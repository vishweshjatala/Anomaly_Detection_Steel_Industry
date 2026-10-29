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
from unet import UNet
from sklearn.decomposition import PCA
from usad import UsadModel
from unet import UNet
from pytorch_tcn import TCN

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001, save_path="best_model.pth"):
        """
        patience: Number of epochs to wait before stopping after no improvement.
        min_delta: Minimum change in validation loss to be considered as improvement.
        save_path: Path to save the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)  # Save the best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs!")
                return True  # Stop training
        return False  # Continue training


class SequenceDataset:
    def __init__(self, df,seq_len):
        self.df = df
        self.seq_len = seq_len
        self.valid_idxs = [(i,i+seq_len) for i in range(0,len(self.df)-seq_len,seq_len)]

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
        start, end = self.valid_idxs[idx]
        values = self.df.iloc[start:end,:]
        indices = self.df.index[start:end]
        assert len(values) == len(indices)
        return {
            'x': torch.from_numpy(values.values).float(),
            'i': indices.to_list()
        }

def collate_fn(batch):
    x = [b['x'] for b in batch]
    i = [b['i'] for b in batch]
    x  = torch.stack(x)
    return {
        'x': x,
        'i': i
    }


class Trainer:
    def __init__(self, config, data, continual=False):
        self.config = config
        self.data = data
        self.continual = continual

        dirname = 'custom_scaler' if self.config.custom_scaler else 'normal_scaler'
        self.save_path = Path(f'./trained_models/{self.config.profile}/{self.config.region}/{dirname}/{self.config.model}')
        if self.continual:
            self.save_path = self.save_path / 'continual'
        self.save_path.mkdir(exist_ok=True,parents=True)

        self.scale()
        self.make_dataloaders()

        if self.config.model == 'lstm' or self.config.model == 'unet':
            if self.config.model == 'lstm':
                self.model = LSTMBasic(num_cols = len(self.data['training_scaled'].columns)).cuda()
            elif self.config.model == 'unet':
                self.model = UNet(num_cols = len(self.data['training_scaled'].columns)).cuda()
            if self.continual:
                if (self.save_path/'model.pt').exists():
                    sd = torch.load(self.save_path/'model.pt',weights_only=True)
                    self.model.load_state_dict(sd)
                    self.config.lr = self.config.continual_lr
                    print('continuing learning from existing weights...')
            print(f'Torch trainable params: {sum([p.numel() for p in self.model.parameters()]):,}',)
            self.train = self.train_lstm

        elif self.config.model == 'pca':
            self.model = PCA(0.98, svd_solver='full')
            print(f'PCA n_components: {self.model.n_components}')
            self.train = self.train_pca

        elif self.config.model == 'usad':
            w_size = self.config.seq_len * len(self.data['training_scaled'].columns) # seq_len * num_fts
            hidden_size = 32
            z_size = self.config.seq_len * hidden_size # seq_len * hidden_size
            self.config.usad_w_size = w_size
            self.config.usad_z_size = z_size
            self.model = UsadModel(w_size=w_size, z_size=z_size).cuda()
            if self.continual:
                if (self.save_path/'model.pt').exists():
                    sd = torch.load(self.save_path/'model.pt',weights_only=True)
                    self.model.load_state_dict(sd)
                    self.config.lr = self.config.continual_lr
                    print('continuing learning from existing weights...')
            print(f'USAD trainable params: {sum([p.numel() for p in self.model.parameters()]):,}',)
            self.train = self.train_usad

    def scale(self):
        if self.config.custom_scaler:
            self.scaler = RegionScaler(region_info=self.data['region_info'],disable_inactive=True)
        else:
            self.scaler = StandardScaler()

        self.scaler.fit(self.data['training'])

        if self.config.custom_scaler:
            self.data['training_scaled'] = self.scaler.transform(self.data['training'])
            self.data['training_scaled'] = self.scaler.mask_off(self.data['training_scaled'])
        else:
            scaled = self.scaler.transform(self.data['training'])
            self.data['training_scaled'] = pd.DataFrame(scaled,columns=self.data['training'].columns,index=self.data['training'].index)

        with open(self.save_path / 'scaler.pkl','wb') as f:
            pickle.dump(self.scaler,f)

    def make_dataloaders(self):
        main_df = self.data['training_scaled']
        diff = main_df.index.to_series().diff()
        # check if the difference is greater than 1 second
        interval_breaks = (diff > pd.Timedelta(seconds=1)).reset_index(drop=True)
        interval_breaks = interval_breaks[interval_breaks==True].index.tolist()
        interval_breaks = [0, *interval_breaks, len(main_df)]
        interval_ranges = [range(a,b-1) for a,b in zip(interval_breaks, interval_breaks[1:])]
        print(f'total {len(interval_ranges)} continuous data sequences')
        dfs = [main_df.iloc[r] for r in tqdm(interval_ranges)]
        datasets = [SequenceDataset(df,seq_len=self.config.seq_len) for df in tqdm(dfs) if len(df)>=self.config.seq_len]
        self.dataloaders = [
            torch.utils.data.DataLoader(ds,shuffle=False,batch_size=self.config.bs, collate_fn=collate_fn)
            for ds in datasets
        ]

    def save(self, dump):
        with open(self.save_path / f"result.pkl",'wb') as f:
            pickle.dump(dump,f)

    def train_pca(self):
        ground_truth_df = self.data['training_scaled']
        self.model.fit(ground_truth_df)
        with open(self.save_path / 'model.pkl','wb') as f:
            pickle.dump(self.model, f)
        encoded = self.model.transform(ground_truth_df)
        decoded = self.model.inverse_transform(encoded)
        prediction_df = pd.DataFrame(decoded,columns=ground_truth_df.columns,index=ground_truth_df.index)
        metrics = pd.DataFrame({'mse': [((prediction_df.values - ground_truth_df).values ** 2).mean()]})

        if self.config.custom_scaler:
            prediction_df[self.scaler.control_signals] = ground_truth_df[self.scaler.control_signals]
            prediction_df = self.scaler.mask_off(prediction_df)

        assert ground_truth_df.shape == prediction_df.shape, f"{ground_truth_df.shape=} {prediction_df.shape=}"

        anomaly_score = pd.DataFrame({'anomaly_score':(prediction_df-ground_truth_df).abs().mean(axis=1).values},index=prediction_df.index)

        dump = {
            'ground_truth': ground_truth_df,
            'predictions': prediction_df,
            'anomaly_score': anomaly_score,
            'config': vars(self.config),
            'metrics': metrics
        }

        self.save(dump)

    def train_lstm(self):

        self.optim = torch.optim.Adam(self.model.parameters(),lr=self.config.lr)
        if self.continual:
            max_lr = 1e-3
            lr_groups = [
                {"params": self.model.enc1.parameters(), "lr": max_lr * 0.1},  # Lower LR
                {"params": self.model.enc2.parameters(), "lr": max_lr * 0.1},
                {"params": self.model.enc3.parameters(), "lr": max_lr * 0.1},
                {"params": self.model.ln1.parameters(), "lr": max_lr * 0.3},   # Intermediate LR
                {"params": self.model.proj.parameters(), "lr": max_lr * 0.5},
                {"params": self.model.dec3.parameters(), "lr": max_lr * 0.8},  # Higher LR
                {"params": self.model.dec2.parameters(), "lr": max_lr * 0.8},
                {"params": self.model.ln2.parameters(), "lr": max_lr * 0.5},
                {"params": self.model.dec1.parameters(), "lr": max_lr},        # Highest LR
            ]
            self.optim = torch.optim.Adam(lr_groups)
        es = EarlyStopping(save_path=self.save_path/'model.pt', patience=self.config.patience)

        losses = []
        for epoch in tqdm(range(self.config.epochs)):
            print(f"{epoch=}")
            total_loss = 0.
            dl_prog = tqdm(self.dataloaders)
            for dl in dl_prog:
                dl_loss = 0.
                for batch in dl:
                    x = batch['x'].cuda()
                    x_hat = self.model(x)
                    loss = nn.functional.mse_loss(x_hat,x)
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    dl_loss += loss.item()
                dl_loss = dl_loss / len(dl)
                dl_prog.set_description(f"{dl_loss:.3f}")
                total_loss += dl_loss
            total_loss = total_loss / len(self.dataloaders)
            stop = es(total_loss, self.model)
            losses.append(total_loss)
            print(f"{epoch=} {total_loss=:.4f}\n")
            if stop:
                print(f'early stopping at {epoch=}')
                break

        metrics = pd.DataFrame({'epoch':list(range(len(losses))),'mse':losses})
        metrics.sort_values(by='mse',inplace=True)

        ground_truth_batches = []
        prediction_batches = []
        index_batches = []
        self.model.eval()

        dl_prog = tqdm(self.dataloaders)
        for dl in dl_prog:
            for batch in dl:
                with torch.no_grad():
                    x = batch['x'].cuda()
                    x_hat = self.model(x)
                    bs, seq_len, _ = x_hat.shape
                    x = x.detach().cpu().reshape(bs*seq_len,-1).numpy()
                    x_hat = x_hat.detach().cpu().reshape(bs*seq_len,-1).numpy()
                    ground_truth_batches.append(x)
                    prediction_batches.append(x_hat)
                    index_batches.append(batch['i'])

        full_index = []
        for idx_batch in index_batches:
            for item in idx_batch:
                full_index.extend(item)

        columns = self.data['training_scaled'].columns
        ground_truth_df = pd.DataFrame(np.concatenate(ground_truth_batches,axis=0),columns=columns,index=full_index)
        prediction_df = pd.DataFrame(np.concatenate(prediction_batches,axis=0),columns=columns,index=full_index)

        if self.config.custom_scaler:
            prediction_df[self.scaler.control_signals] = ground_truth_df[self.scaler.control_signals]
            prediction_df = self.scaler.mask_off(prediction_df)

        assert ground_truth_df.shape == prediction_df.shape, f"{ground_truth_df.shape=} {prediction_df.shape=}"

        anomaly_score = pd.DataFrame({'anomaly_score':(prediction_df-ground_truth_df).abs().mean(axis=1).values},index=prediction_df.index)

        dump = {
            'ground_truth': ground_truth_df,
            'predictions': prediction_df,
            'anomaly_score': anomaly_score,
            'config': vars(self.config),
            'metrics': metrics
        }

        self.save(dump)

    def train_usad(self):

        optimizer1 = torch.optim.Adam(list(self.model.encoder.parameters())+list(self.model.decoder1.parameters()),lr=self.config.lr)
        optimizer2 = torch.optim.Adam(list(self.model.encoder.parameters())+list(self.model.decoder2.parameters()),lr=self.config.lr)

        es = EarlyStopping(save_path=self.save_path/'model.pt', patience=self.config.patience)

        losses1,losses2 = [],[]
        for epoch in tqdm(range(self.config.epochs)):
            print(f"{epoch=}")
            total_loss1 = 0.
            total_loss2 = 0.
            dl_prog = tqdm(self.dataloaders)

            torch.autograd.set_detect_anomaly(True)
            for dl in dl_prog:
                dl_loss1 = 0.
                dl_loss2 = 0.
                for batch in dl:
                    x = batch['x'].cuda()
                    x = x.flatten(1,-1)
                    w1,w2,w3 = self.model(x)
                    n = epoch + 1
                    loss1 = 1/n*torch.mean((x-w1)**2)+(1-1/n)*torch.mean((x-w3)**2)
                    loss2 = 1/n*torch.mean((x-w2)**2)-(1-1/n)*torch.mean((x-w3)**2)

                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    loss1.backward(retain_graph=True)
                    loss2.backward()

                    optimizer1.step()
                    optimizer2.step()

                    dl_loss1 += loss1.item()
                    dl_loss2 += loss2.item()

                dl_loss1 = dl_loss1 / len(dl)
                dl_loss2 = dl_loss2 / len(dl)
                dl_prog.set_description(f"{dl_loss1:.3f} | {dl_loss2:.3f}")
                total_loss1 += dl_loss1
                total_loss2 += dl_loss2
            total_loss1 = total_loss1 / len(self.dataloaders)
            total_loss2 = total_loss2 / len(self.dataloaders)

            stop = es(total_loss2, self.model)
            losses1.append(total_loss1)
            losses2.append(total_loss2)
            print(f"{epoch=} {total_loss1=:.4f} {total_loss2=:.4f}\n")
            if stop:
                print(f'early stopping at {epoch=}')
                break

        metrics = pd.DataFrame({'reconstruction': losses1, 'adversarial': losses2})

        all_batch_scores = []
        index_batches = []
        self.model.eval()

        dl_prog = tqdm(self.dataloaders)
        for dl in dl_prog:
            for batch in dl:
                with torch.no_grad():
                    x = batch['x'].cuda()
                    x = x.flatten(1,-1)
                    w1,w2,w3 = self.model(x)
                    batch_scores = self.config.usad_alpha*torch.mean((x-w1)**2,axis=1)+self.config.usad_beta*torch.mean((x-w2)**2,axis=1)
                    batch_scores = batch_scores.repeat(self.config.seq_len,1).T
                    batch_scores = batch_scores.cpu().numpy().tolist()

                    all_batch_scores.append(batch_scores)
                    index_batches.append(batch['i'])

        full_index = []
        full_scores = []
        for idx_batch,idx_score in zip(index_batches,all_batch_scores):
            for idx,score in zip(idx_batch,idx_score):
                full_index.extend(idx)
                full_scores.extend(score)

        assert len(full_index) == len(full_scores), f'{len(full_index)=} | {len(full_scores)=}'

        anomaly_scores = pd.DataFrame({'anomaly_score': full_scores}, index=full_index)

        dump = {
            'ground_truth': self.data['training_scaled'],
            'anomaly_score': anomaly_scores,
            'config': vars(self.config),
            'metrics': metrics
        }

        self.save(dump)


class Evaluator:
    def __init__(self, model_dir, cobble_removed_df):

        with open(model_dir / 'result.pkl','rb') as f:
            result = pickle.load(f)
            self.config = result['config']
            self.clean_df_anomaly_scores = result['anomaly_score']

        with open(model_dir/'scaler.pkl','rb') as f:
            self.scaler = pickle.load(f)
        self.cobble_removed_df = cobble_removed_df
        if self.config['custom_scaler']:
            self.scaled_eval_df = self.scaler.transform(self.cobble_removed_df)
            self.scaled_eval_df = self.scaler.mask_off(self.scaled_eval_df)
        else:
            scaled_eval_values = self.scaler.transform(self.cobble_removed_df)
            cols, idxs = self.cobble_removed_df.columns, self.cobble_removed_df.index
            self.scaled_eval_df = pd.DataFrame(scaled_eval_values, index=idxs, columns=cols)

        cols = self.scaled_eval_df.columns

        if self.config['model'] in ['lstm', 'usad','unet']:
            model_wts = torch.load(model_dir / 'model.pt', weights_only=True)
            self.make_dataloaders()
            if self.config['model'] == 'lstm' or self.config['model'] == 'unet':
                if self.config['model'] == 'lstm':
                    self.model = LSTMBasic(num_cols = len(cols)).cuda()
                elif self.config['model'] == 'unet':
                    self.model = UNet(num_cols = len(cols)).cuda()
                self.model.load_state_dict(model_wts)
                self.eval = self.eval_lstm
            if self.config['model'] == 'usad':
                self.model = UsadModel(w_size=self.config['usad_w_size'],z_size=self.config['usad_z_size']).cuda()
                self.model.load_state_dict(model_wts)
                self.eval = self.eval_usad
            self.model.eval()
        elif self.config['model'] == 'pca':
            with open(model_dir/'model.pkl','rb') as f:
                self.model = pickle.load(f)
                self.eval = self.eval_pca


    def make_dataloaders(self):
        main_df = self.scaled_eval_df
        diff = main_df.index.to_series().diff()
        # check if the difference is greater than 1 second
        interval_breaks = (diff > pd.Timedelta(seconds=1)).reset_index(drop=True)
        interval_breaks = interval_breaks[interval_breaks==True].index.tolist()
        interval_breaks = [0, *interval_breaks, len(main_df)]
        interval_ranges = [range(a,b-1) for a,b in zip(interval_breaks, interval_breaks[1:])]
        print(f'total {len(interval_ranges)} continuous data sequences')
        dfs = [main_df.iloc[r] for r in tqdm(interval_ranges)]
        datasets = [SequenceDataset(df,seq_len=self.config['seq_len']) for df in tqdm(dfs)] # not for eval: if len(df)>=self.config['seq_len']]
        self.dataloaders = [
            torch.utils.data.DataLoader(ds,shuffle=False,batch_size=self.config['bs'], collate_fn=collate_fn)
            for ds in datasets
        ]

    def get_train_stats(self):
        results = dict(self.clean_df_anomaly_scores['anomaly_score'].describe())
        results['90%'] = np.percentile(self.clean_df_anomaly_scores['anomaly_score'].values,90)
        results['95%'] = np.percentile(self.clean_df_anomaly_scores['anomaly_score'].values,95)
        results['99%'] = np.percentile(self.clean_df_anomaly_scores['anomaly_score'].values,99)
        return results

    def eval_pca(self):
        encoded = self.model.transform(self.scaled_eval_df)
        decoded = self.model.inverse_transform(encoded)

        prediction_df = pd.DataFrame(decoded,columns=self.scaled_eval_df.columns,index=self.scaled_eval_df.index)

        if self.config['custom_scaler']:
            prediction_df[self.scaler.control_signals] = self.scaled_eval_df[self.scaler.control_signals]
            prediction_df = self.scaler.mask_off(prediction_df)

        anomaly_scores = np.abs(self.scaled_eval_df.values-decoded).mean(axis=1).reshape(-1)
        return pd.DataFrame({'anomaly_score': anomaly_scores},index=self.scaled_eval_df.index)

    def eval_lstm(self):
        ground_truth_batches = []
        prediction_batches = []
        index_batches = []

        dl_prog = tqdm(self.dataloaders)
        for dl in dl_prog:
            for batch in dl:
                with torch.no_grad():
                    x = batch['x'].cuda()
                    x_hat = self.model(x)
                    bs, seq_len, _ = x_hat.shape
                    x = x.detach().cpu().reshape(bs*seq_len,-1).numpy()
                    x_hat = x_hat.detach().cpu().reshape(bs*seq_len,-1).numpy()
                    ground_truth_batches.append(x)
                    prediction_batches.append(x_hat)
                    index_batches.append(batch['i'])

        full_index = []
        for idx_batch in index_batches:
            for item in idx_batch:
                full_index.extend(item)

        columns = self.scaled_eval_df.columns
        ground_truth_df = pd.DataFrame(np.concatenate(ground_truth_batches,axis=0),columns=columns,index=full_index)
        prediction_df = pd.DataFrame(np.concatenate(prediction_batches,axis=0),columns=columns,index=full_index)

        if self.config['custom_scaler']:
            prediction_df[self.scaler.control_signals] = ground_truth_df[self.scaler.control_signals]
            prediction_df = self.scaler.mask_off(prediction_df)

        assert ground_truth_df.shape == prediction_df.shape, f"{ground_truth_df.shape=} {prediction_df.shape=}"

        anomaly_score = pd.DataFrame({'anomaly_score':(prediction_df-ground_truth_df).abs().mean(axis=1).values},index=prediction_df.index)
        return anomaly_score


    def eval_usad(self):
        all_batch_scores = []
        index_batches = []

        dl_prog = tqdm(self.dataloaders)
        for dl in dl_prog:
            for batch in dl:
                with torch.no_grad():
                    x = batch['x'].cuda()
                    x = x.flatten(1,-1)
                    w1,w2,w3 = self.model(x)
                    batch_scores = self.config['usad_alpha']*torch.mean((x-w1)**2,axis=1)+self.config['usad_beta']*torch.mean((x-w2)**2,axis=1)
                    batch_scores = batch_scores.repeat(self.config['seq_len'],1).T
                    batch_scores = batch_scores.cpu().numpy().tolist()

                    all_batch_scores.append(batch_scores)
                    index_batches.append(batch['i'])

        full_index = []
        full_scores = []
        for idx_batch,idx_score in zip(index_batches,all_batch_scores):
            for idx,score in zip(idx_batch,idx_score):
                full_index.extend(idx)
                full_scores.extend(score)

        assert len(full_index) == len(full_scores), f'{len(full_index)=} | {len(full_scores)=}'

        anomaly_scores = pd.DataFrame({'anomaly_score': full_scores}, index=full_index)

        return anomaly_scores



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--profile',required=True,choices=['12mm','16mm'])
    parser.add_argument('--region',required=True,choices=['Stand_1-6','Stand_7-12','Stand_13-18','FFB_L1','FFB_L2'])
    parser.add_argument('--model',required=True,choices=['lstm','pca','usad','unet'])
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--seq_len',type=int,default=30)
    parser.add_argument('--bs',type=int,default=128)
    parser.add_argument('--custom_scaler',action='store_true')
    parser.add_argument('--usad_alpha',type=float,default=0.5)
    parser.add_argument('--usad_beta',type=float,default=0.5)
    parser.add_argument('--patience',type=int,default=15)

    args = parser.parse_args()
    print(args)
    data = get_data(profile=args.profile,limit_to_known=False,region=args.region)
    trainer = Trainer(config=args,data=data)
    trainer.train()