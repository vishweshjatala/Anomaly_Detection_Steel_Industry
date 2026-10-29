import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm
from preprocess import get_data
import torch
import torch.nn as nn
from scaler import RegionScaler, ConditionalScaler, StandardScaler
from pathlib import Path
import argparse
from lstm_basic import LSTMBasic
from sklearn.decomposition import PCA
from usad import UsadModel
from trainer import Trainer, Evaluator
from tuner import run_hyperparameter_tuning

def remove_leading_time(df, cobble_intervals_df, lead=7):
    keep_mask = pd.Series(True, index=df.index)

    for start_time in pd.to_datetime(cobble_intervals_df['start']):
        # Calculate the window to remove
        window_start = start_time - pd.Timedelta(minutes=lead)
        window_end = start_time - pd.Timedelta(seconds=1)

        cobble_intervals_df['anomaly_start_time'] = window_start
        cobble_intervals_df['anomaly_end_time'] = window_end

        # Update mask to exclude this window
        window_mask = (df.index >= window_start) & (df.index <= window_end)
        keep_mask = keep_mask & ~window_mask

    # Apply the mask
    filtered_df = df[keep_mask].copy()

    return filtered_df

def get_chunks(df, cobble_intervals):
    groups = {start: group for start, group in df.groupby(pd.Grouper(freq='3h'))}
    chunks = [df for df in groups.values()][2:]
    merged_chunks = [chunks[0]]
    for i in range(1,len(chunks)):
        if len(chunks[i]) < 500:
            prev_chunk = merged_chunks[-1]
            merged_chunks.pop()
            new_chunk = pd.concat([prev_chunk, chunks[i]])
            merged_chunks.append(new_chunk)
        else:
            merged_chunks.append(chunks[i])

    cobble_intervals = cobble_intervals
    all_chunks = {chunk_id: None for chunk_id in range(len(merged_chunks))}
    for chunk_id,chunk in enumerate(merged_chunks):
        chunked_data = dict()
        start,end = chunk.index[0], chunk.index[-1]
        failure_df = cobble_intervals[(cobble_intervals['start'] >= start) & (cobble_intervals['start'] < end)].reset_index(drop=True)
        chunked_data['cobble_removed'] = chunk
        chunked_data['cobble_intervals'] = failure_df
        chunked_data['training'] = remove_leading_time(chunk, failure_df)
        all_chunks[chunk_id] = chunked_data

    return all_chunks


def dynamic_thresholding(trainer, eval_chunk):
    normal_data = eval_chunk['cobble_removed']
    start = normal_data.index[0]
    end = normal_data.index[0]+pd.Timedelta(minutes=15)
    calib_data = normal_data[(normal_data.index>=start) & (normal_data.index<=end)]
    scaler = trainer.scaler
    config = trainer.config

    if config.custom_scaler:
        calib_data_scaled = scaler.transform(calib_data)
        calib_data_scaled = scaler.mask_off(calib_data_scaled)
    else:
        scaled = scaler.transform(calib_data)
        calib_data_scaled = pd.DataFrame(scaled,columns=calib_data.columns,index=calib_data.index)

    mean = calib_data_scaled.values.mean()
    std = calib_data_scaled.values.std()
    print(len(calib_data), mean, std)
    threshold = mean + 1.1 * std

    return threshold



def run_continual_learning(all_chunks, area_data, config):

    best_metrics = []

    print('running continual learning...')

    chunk_ids = list(all_chunks.keys())
    for idx in range(0,len(chunk_ids)-1):
        chunk_id = chunk_ids[idx]
        if chunk_id == 0:
            train_chunk = all_chunks[0]
            eval_chunks = [(0,train_chunk), (1,all_chunks[1])]
        else:
            train_chunk = all_chunks[chunk_id]
            eval_chunks = [(chunk_id+1,all_chunks[chunk_id+1])]

        train_chunk['region_info'] = area_data['region_info']
        train_chunk['columns'] = area_data['columns']
        if chunk_id == 0:
            config.epochs = 100
        trainer = Trainer(config=config,data=train_chunk,continual=True)
        trainer.train()
        model_dir = trainer.save_path
        all_chunks[chunk_id]['model_dir'] = model_dir
        all_chunks[chunk_id]['eval'] = dict()

        for eval_id, eval_chunk in tqdm(eval_chunks):
            print(f"{eval_id=}", train_chunk['training'].shape, eval_chunk['cobble_removed'].shape)
            try:
                evaluator = Evaluator(model_dir=model_dir, cobble_removed_df=eval_chunk['cobble_removed'])
            except Exception as e:
                print(e)
                eval_chunk['region_info'] = area_data['region_info']
                eval_chunk['columns'] = area_data['columns']
                trainer = Trainer(config=config,data=eval_chunk,continual=True)
                trainer.train()
                evaluator = Evaluator(model_dir=model_dir, cobble_removed_df=eval_chunk['cobble_removed'])
            cobble_anomaly_scores = evaluator.eval()
            all_chunks[chunk_id]['eval'][eval_id] = {
                'anomaly_scores': cobble_anomaly_scores,
                'clean_stats': evaluator.get_train_stats()
            }

            # new_threshold = dynamic_thresholding(trainer, eval_chunk)
            # print('new_threshold',new_threshold)

            tune_metrics = run_hyperparameter_tuning(
                cobble_anomaly_scores,
                eval_chunk['cobble_intervals'],
                evaluator.get_train_stats(),
                max_lead_time=7,
                threshold_mode='z-score',
                continual=True,
                continual_threshold=None
                )
            best = tune_metrics.sort_values(by='f1_score',ascending=False).iloc[0].to_dict()
            best_metrics.append(best)

    res = {
        'config': vars(config),
        'metrics': best_metrics
    }
    with open(model_dir / 'continual_results.pkl', 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--profile',required=True,choices=['12mm','16mm'])
    parser.add_argument('--region',required=True,choices=['Stand_1-6','Stand_7-12','Stand_13-18','FFB_L1','FFB_L2'])
    parser.add_argument('--model',required=True,choices=['lstm','pca','usad','lstm_attn','tcn'])
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--seq_len',type=int,default=30)
    parser.add_argument('--bs',type=int,default=32)
    parser.add_argument('--custom_scaler',action='store_true')
    parser.add_argument('--usad_alpha',type=float,default=0.5)
    parser.add_argument('--usad_beta',type=float,default=0.5)
    parser.add_argument('--patience',type=int,default=7)

    args = parser.parse_args()

    data = get_data(profile=args.profile,region=args.region,limit_to_known=False)

    args.continual_lr = 5e-4

    area_data = {
        'region_info': data['region_info'],
        'columns': data['columns']
    }

    chunks = get_chunks(data['cobble_removed'], cobble_intervals=data['cobble_intervals'])

    run_continual_learning(chunks, area_data, args)