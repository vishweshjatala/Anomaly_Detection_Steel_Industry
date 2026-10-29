import numpy as np
import pandas as pd
from preprocess import get_data
from types import SimpleNamespace
from pathlib import Path
from trainer import Evaluator
from tqdm.auto import tqdm
from itertools import product
import concurrent.futures
import argparse
import traceback

class GroupingAlgorithm:
    def __init__(self, threshold, k, td, col='anomaly_score'):
        self.threshold = threshold
        self.k = k
        self.td = td
        self.col = col
        self.reset()

    def reset(self):
        self.count = 0
        self.prev_time = None

    def __call__(self, idx, row):
        score = row[self.col]
        if score >= self.threshold:
            if self.prev_time == None:
                self.count += 1
                self.prev_time = idx
            else:
                if (idx-self.prev_time).total_seconds() <= self.td:
                    self.count += 1
                    self.prev_time = idx
                    if self.count == self.k:
                        prediction_time = self.prev_time
                        self.reset()
                        return True, prediction_time
                else:
                    self.reset()

        return False, None


def get_predictions(anomaly_scores, threshold, k, td):
    ga = GroupingAlgorithm(threshold=threshold,k=k,td=td)
    predictions = []
    for idx, row in anomaly_scores.iterrows():
        res,pred_time = ga(idx, row)
        if res:
            predictions.append(pred_time)
    return predictions

def get_metrics(cobble_intervals, prediction_times, max_lead_time=7):

    metrics = {
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'avg_max_lead_time': 0,
        'avg_min_lead_time': 0,
        'max_lead_time': 0,
        'min_lead_time': 0
    }
    if len(prediction_times) == 0:
        return metrics

    actual_times = {actual: dict() for actual in list(cobble_intervals['start'])}

    for idx, actual in enumerate(actual_times.keys()):
        min_time = actual - pd.Timedelta(minutes=max_lead_time)
        max_time = actual - pd.Timedelta(seconds=30)
        true_predictions = list(filter(lambda t:t>=min_time and t<=max_time, prediction_times))
        actual_times[actual]['prediction_times'] = true_predictions
        actual_times[actual]['lead_times'] = list(map(lambda t:(max_time-t).total_seconds(),true_predictions))
        actual_times[actual]['max_lead_time'] = max(actual_times[actual]['lead_times']) if len(actual_times[actual]['lead_times'])> 0 else None
        actual_times[actual]['min_lead_time'] = min(actual_times[actual]['lead_times']) if len(actual_times[actual]['lead_times'])> 0 else None
        avg_lead_time = sum(actual_times[actual]['lead_times']) / len(actual_times[actual]['lead_times']) if len(actual_times[actual]['lead_times'])> 0 else 0
        actual_times[actual]['avg_lead_time'] = avg_lead_time

    true_positives = sum([len(actual_times[a]['prediction_times']) for a in actual_times if len(actual_times[a]['prediction_times'])>0])
    false_positives = len(prediction_times) - true_positives
    false_negatives = len([len(actual_times[a]['prediction_times']) for a in actual_times if len(actual_times[a]['prediction_times'])==0])


    if true_positives == 0:
        return metrics

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = (2*precision*recall)/(precision+recall)
    false_negative_rate = false_negatives / (true_positives + false_negatives)

    max_lead_times = [actual_times[a]['max_lead_time'] for a in actual_times if len(actual_times[a]['prediction_times'])>0]
    min_lead_times = [actual_times[a]['min_lead_time'] for a in actual_times if len(actual_times[a]['prediction_times'])>0]
    avg_lead_times = [actual_times[a]['avg_lead_time'] for a in actual_times if len(actual_times[a]['prediction_times'])>0]

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'miss_rate': false_negative_rate,
        'avg_max_lead_time': sum(max_lead_times) / len(max_lead_times),
        'avg_min_lead_time': sum(min_lead_times) / len(min_lead_times),
        'avg_lead_time': sum(avg_lead_times) / len(avg_lead_times),
        'max_lead_time': max(max_lead_times),
        'min_lead_time': min(min_lead_times),
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives
    }

    return metrics


def run_combination(anomaly_scores, cobble_intervals, max_lead_time, threshold, k, td):
    # print(f'running combination: {threshold=} {k=} {td=}')
    predictions = get_predictions(anomaly_scores, threshold, k, td)
    metrics = get_metrics(cobble_intervals=cobble_intervals,prediction_times=predictions, max_lead_time=max_lead_time)
    metrics.update({'threshold': threshold, 'k':k, 'td': td})
    return metrics


def run_hyperparameter_tuning(anomaly_scores, cobble_intervals, stats, max_lead_time, threshold_mode='z-score', continual=False, continual_threshold=None):

    diff = 0.2 if continual else 0.1

    if threshold_mode == 'z-score':
        mean,std = stats['mean'], stats['std']
        start = mean + 2.5 * std
        end = mean + 5 * std

    elif threshold_mode == 'pct':
        start = stats['95%']
        end = stats['99%']

    thresholds = [start]
    while start <= end:
        th = start + diff
        thresholds.append(th)
        start = th

    td = [2,3,4,5,6,7,8,9]
    k = [3,4,5,6,7,8,9]
    combinations = list(product(thresholds, k, td))
    print('total combinations', len(combinations))

    metrics = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:

        futures = []
        for th, k, td in combinations:
            comb = (anomaly_scores, cobble_intervals, max_lead_time, th, k, td)
            futures.append(executor.submit(run_combination, *comb))

        for future in concurrent.futures.as_completed(futures):  # Process results as they complete
            with tqdm(total=len(combinations), desc="hyperparameter tuning") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        metrics.append(result)
                        pbar.update(1)  # Update the progress bar
                    except Exception as e:
                        print(f"An error occurred: {e}\n{traceback.format_exc()}")
                        pbar.update(1)

    metrics = pd.DataFrame(metrics)
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--profile',required=True,choices=['12mm','16mm'])
    parser.add_argument('--region',required=True,choices=['Stand_1-6','Stand_7-12','Stand_13-18','FFB_L1','FFB_L2'])
    parser.add_argument('--model',required=True,choices=['lstm','pca','usad','unet'])
    parser.add_argument('--custom_scaler',action='store_true')
    parser.add_argument('--threshold',choices=['z-score','pct'],default='z-score')

    args = parser.parse_args()

    dirname = 'custom_scaler' if args.custom_scaler else 'normal_scaler'
    model_dir = Path(f'./trained_models/{args.profile}/{args.region}/{dirname}/{args.model}')

    if not model_dir.exists():
        print('train the model first')
    else:
        data = get_data(profile=args.profile,limit_to_known=False,region=args.region)
        evaluator = Evaluator(model_dir, data['cobble_removed'])
        cobble_anomaly_scores = evaluator.eval()
        clean_anomaly_score_stats = evaluator.get_train_stats()

        args.threshold = 'pct' if args.model=='usad' else 'z-score'
        print(args)
        metrics = run_hyperparameter_tuning(
            anomaly_scores=cobble_anomaly_scores,
            cobble_intervals=data['cobble_intervals'],
            stats=clean_anomaly_score_stats,
            max_lead_time=7,
            threshold_mode=args.threshold
        )
        metrics.to_csv(model_dir/'hyperparameters.csv',index=False)