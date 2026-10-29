from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

__all__ = ['ConditionalScaler','RegionScaler']

class ConditionalScaler(BaseEstimator, TransformerMixin):
    def __init__(self, column, control_column):
        self.column = column
        self.control_column = control_column
        self.scaler_off = StandardScaler()
        self.scaler_on = StandardScaler()

    def fit(self, X, y=None):
        # Split data based on control signal
        mask_0 = X[self.control_column] == 0
        mask_1 = X[self.control_column] == 1

        # Fit separate scalers for each condition
        if mask_0.any():
            self.scaler_off.fit(X[mask_0].drop(columns=[self.control_column]))
        if mask_1.any():
            self.scaler_on.fit(X[mask_1].drop(columns=[self.control_column]))
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Transform data based on control signal
        mask_0 = X_copy[self.control_column] == 0
        mask_1 = X_copy[self.control_column] == 1

        if mask_0.any():
            X_copy.loc[mask_0, X_copy.columns != self.control_column] = self.scaler_off.transform(
                X_copy[mask_0].drop(columns=[self.control_column]))
        if mask_1.any():
            X_copy.loc[mask_1, X_copy.columns != self.control_column] = self.scaler_on.transform(
                X_copy[mask_1].drop(columns=[self.control_column]))

        return X_copy.drop(columns=[self.control_column])


class RegionScaler(object):
    def __init__(self, region_info, disable_inactive=False):
        self.region_info = region_info
        self.equipment_scalers = dict()
        self.disable_inactive = disable_inactive
        for equipment_name in list(self.region_info['Equipment'].keys()):
            self._create_column_transformer(
                equipment_name=equipment_name,
                equipment_info=self.region_info['Equipment'][equipment_name]
                )

    def _create_column_transformer(self, equipment_name, equipment_info):
        transformers = []
        columns = []

        # Add conditional scalers for affected columns
        if equipment_info.get('affected_signals',None) is not None:
            for col in equipment_info['affected_signals']:
                transformers.append(
                    (f'conditional_scaler_{col}',
                    ConditionalScaler(column=col,control_column=equipment_info['control_signal']),
                    [col, equipment_info['control_signal']])
                )
                columns.append(col)

        # Add regular scalers for unaffected columns
        if equipment_info.get('unaffected_signals',None) is not None:
            for col in equipment_info['unaffected_signals']:
                transformers.append(
                    (f'standard_scaler_{col}',
                    StandardScaler(),
                    [col])
                )
                columns.append(col)

        self.equipment_scalers[equipment_name] = {
            'transformer': ColumnTransformer(
                            transformers=transformers,
                            remainder='drop'
                        ),
            'columns': columns
        }

    def fit(self, X):
        all_columns = list()
        control_signals = list()
        for equipment_name in self.equipment_scalers:
            self.equipment_scalers[equipment_name]['transformer'].fit(X)
            main_cols = self.equipment_scalers[equipment_name]['columns']
            equipment_info=self.region_info['Equipment'][equipment_name]
            if equipment_info.get('affected_signals',None) is not None:
                control_signal = self.region_info['Equipment'][equipment_name]['control_signal']
                all_columns.extend([*main_cols, control_signal])
                control_signals.append(control_signal)
            else:
                all_columns.extend(main_cols)
        self.columns = all_columns
        self.control_signals = control_signals
        # print(self.columns)


    def transform(self, X):
        X_copy = X.copy()
        index = X_copy.index
        transformed_dfs = []
        for equipment_name in self.region_info['Equipment']:
            equipment_scaler_info = self.equipment_scalers[equipment_name]
            scaler, columns = equipment_scaler_info['transformer'], equipment_scaler_info['columns']
            equipment_info=self.region_info['Equipment'][equipment_name]
            scaled_df = pd.DataFrame(scaler.transform(X_copy),index=index,columns=columns)
            if equipment_info.get('affected_signals',None) is not None:
                control_signal = self.region_info['Equipment'][equipment_name]['control_signal']
                scaled_df[control_signal] = X_copy[control_signal].copy()

            # STAGE 1: CONSIDER ACTIVE AREA ONLY FOR AFFECTED SIGNALS
            if self.disable_inactive:
                if equipment_info.get('affected_signals',None) is not None:
                    for signal in self.region_info['Equipment'][equipment_name]['affected_signals']:
                        scaled_df[signal] = scaled_df[signal] * scaled_df[control_signal]
                        # scaled_df[scaled_df[signal]==-0.] = 0.

            transformed_dfs.append(scaled_df)

        transformed_df = pd.concat(transformed_dfs,axis=1)
        transformed_df = transformed_df.loc[:,~transformed_df.columns.duplicated()].copy()
        # print(transformed_df.columns,self.columns)
        transformed_df = transformed_df[self.columns]
        # assert len(transformed_df.columns) == len(X.columns), 'oh no'
        # transformed_df = transformed_df[X.columns]
        return transformed_df

    def get_0_to_1(self,signal):
        shifted = signal-signal.shift(1)
        pos_edges = shifted.where(shifted==1).dropna().index
        if not pos_edges.empty:
            return pos_edges.to_list()

    def get_1_to_0(self,signal):
        shifted = signal-signal.shift(1)
        neg_edges = shifted.where(shifted==-1).dropna().index
        if not neg_edges.empty:
            return neg_edges.to_list()

    def mask_off(self, predictions, clip_transitions=False):
        """
        only applicable when self.disable_inactive is set to True
        when we get reconstructed output logits from the autoencoder,
        we also need to mask the affected signals where the control signal is zero.
        """
        if not self.disable_inactive:
            print('why are you ruining your life')
            return predictions

        if clip_transitions:
            for equipment_name in self.region_info['Equipment']:
                equipment_info=self.region_info['Equipment'][equipment_name]
                if equipment_info.get('affected_signals',None) is not None:
                    control_signal = self.region_info['Equipment'][equipment_name]['control_signal']
                    control_signal_values = predictions.loc[:,control_signal]
                    z_to_o_list = self.get_0_to_1(control_signal_values)
                    if z_to_o_list is not None:
                        for z_to_o in z_to_o_list:
                            end = z_to_o + pd.Timedelta(seconds=5)
                            predictions.loc[(predictions.index>=z_to_o)&(predictions.index<=end),control_signal] = 1.
                    o_to_z_list = self.get_1_to_0(control_signal_values)
                    if o_to_z_list is not None:
                        for o_to_z in o_to_z_list:
                            start = o_to_z - pd.Timedelta(seconds=5)
                            predictions.loc[(predictions.index>=start)&(predictions.index<=o_to_z),control_signal] = 1.



        for equipment_name in self.region_info['Equipment']:
            if self.disable_inactive:
                equipment_info=self.region_info['Equipment'][equipment_name]
                if equipment_info.get('affected_signals',None) is not None:
                    control_signal = self.region_info['Equipment'][equipment_name]['control_signal']
                    for signal in self.region_info['Equipment'][equipment_name]['affected_signals']:
                        predictions[signal] = predictions[signal] * predictions[control_signal]
                        # predictions[predictions[signal]==-0.] = 0.

        return predictions