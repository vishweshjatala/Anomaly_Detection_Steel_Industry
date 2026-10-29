import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
import yaml
import pickle
from pathlib import Path

def load_data(profile='16mm'):
    base_path = Path('/data2/shreyas/bsp/DATA/iba_data_1s/iba_data_for_training') / profile
    files = list(base_path.glob('*.parquet'))
    data = pd.concat([pd.read_parquet(f) for f in tqdm(files)])
    data = data.sort_index()
    return data


def remove_loc_mat_presence_zero(df):
    mat_presence_signal = '[9.101]'
    print('mat_presence')
    print(df[mat_presence_signal].value_counts())
    df_removed_mp0 = df[df[mat_presence_signal]!=0]
    print(df_removed_mp0[mat_presence_signal].value_counts(dropna=False))

    return df_removed_mp0


def remove_ghost_rolling(df):
    ghost_rolling_signal = '[9.226]'
    print('ghost_rolling')
    print(df[ghost_rolling_signal].value_counts())
    df_removed_gh = df[df[ghost_rolling_signal]!=1]
    print(df_removed_gh[ghost_rolling_signal].value_counts(dropna=False))
    return df_removed_gh

def get_cobble_intervals(df):

    print('getting permissive intervals')
    copy = df.copy().reset_index(names='time')
    permissive_intervals_signal = '[9.74]'

    copy['group'] = (copy[permissive_intervals_signal].ne(0) | copy[permissive_intervals_signal].shift(1).ne(0)).cumsum()

    start_times = copy[copy[permissive_intervals_signal] == 0].groupby('group')['time'].min()
    end_times = copy[copy[permissive_intervals_signal] == 0].groupby('group')['time'].max()

    permissive_intervals_df = pd.DataFrame({'start': start_times, 'end': end_times})
    permissive_intervals_df['duration[min]'] = (permissive_intervals_df['end']-permissive_intervals_df['start']).dt.total_seconds() / 60.

    chops = ['[21.89]','[21.90]'] # SH1_Chop, SH2_Chop
    timestamps_df = []

    #calculate timetamps when (SH2_Chop Req main flag OB1) or (SH1_Chop Req main flag OB1 is high)
    for i in tqdm(range(len(df) - 1)):
        if (df[chops[0]].iloc[i] == 0 and df[chops[0]].iloc[i + 1] == 1) or \
           (df[chops[1]].iloc[i] == 0 and df[chops[1]].iloc[i + 1] == 1) :
            timestamps_df.append(df.index[i + 1])

    timestamps_df = pd.DataFrame({'timestamp': timestamps_df})

    print('calculating when the cobble happened via signals')
    cobble_df = {
        'start': [],
        'end': [],
        'duration[min]': []
    }
    for _, row in tqdm(permissive_intervals_df.iterrows()):
        for _, row1 in timestamps_df.iterrows():
            if row1['timestamp'] >= row['start'] and row1['timestamp'] <= row['end']:
                cobble_df['start'].append(row['start'])
                cobble_df['end'].append(row['end'])
                cobble_df['duration[min]'].append(row['duration[min]'])
    cobble_df = pd.DataFrame(cobble_df)
    cobble_df.drop_duplicates(ignore_index=True, inplace=True)
    cobble_df_filtered = cobble_df[cobble_df['duration[min]'] >= 4].reset_index(drop=True)

    return cobble_df_filtered

def remove_cobble_intervals_from_data(df, cobble_intervals):

    mask = pd.Series(False, index=df.index)
    for _, row in cobble_intervals.iterrows():
        start_time = row['start']
        end_time = row['end']
        mask |= (df.index >= start_time) & (df.index <= end_time)

    # Invert the mask to keep rows not in the mask
    df = df[~mask]

    print('removed durations when the cobble had occurred (cobble actual start) TILL (mill restarted after incident)\n')
    return df

def remove_leading_time(df, cobble_intervals_df, lead=7):


    starts = cobble_intervals_df['start']
    print(starts)
    # Initialize mask as all True
    keep_mask = pd.Series(True, index=df.index)

    # Process each interval
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

    return filtered_df, cobble_intervals_df


def get_anomaly_df(data_df, anomaly_df):
    """
    Filter rows from data_df that fall within any of the time ranges
    specified in anomaly_df.

    Parameters:
    -----------
    data_df : pandas.DataFrame
        DataFrame containing the data to filter. Must have a datetime column.
    anomaly_df : pandas.DataFrame
        DataFrame containing anomaly_start_time and anomaly_end_time columns

    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame containing only rows that fall within any of the
        anomaly time ranges
    """
    # Create an empty mask
    mask = pd.Series(False, index=data_df.index)

    # For each anomaly period, update the mask
    for _, row in anomaly_df.iterrows():
        start = row['anomaly_start_time']
        end = row['anomaly_end_time']
        period_mask = (data_df.index >= start) & (data_df.index <= end)
        mask = mask | period_mask

    # Apply the mask to get the filtered DataFrame
    anomaly_df = data_df[mask].copy()

    return anomaly_df


def preprocess(df):
    df_mill_on = remove_loc_mat_presence_zero(df)
    df_mill_on = remove_ghost_rolling(df_mill_on)
    cobble_intervals = get_cobble_intervals(df)
    cobble_removed_df = remove_cobble_intervals_from_data(df=df_mill_on, cobble_intervals=cobble_intervals)
    filtered_df, cobble_intervals = remove_leading_time(cobble_removed_df, cobble_intervals, lead=7)
    cobble_intervals['anomaly_start_time'] = cobble_intervals['start'] - pd.Timedelta(minutes=7)
    cobble_intervals['anomaly_end_time'] = cobble_intervals['start'] - pd.Timedelta(seconds=1)
    anomaly_df = get_anomaly_df(df_mill_on, cobble_intervals)
    return df_mill_on, cobble_removed_df, filtered_df, cobble_intervals, anomaly_df

def load_region(region_name):
    yaml_path = Path("/data2/shreyas/bsp/V3/REGIONS") / f"{region_name}.yaml"
    with open(yaml_path) as stream:
        data = yaml.safe_load(stream)
    equipment_signals = [list(s.values()) for s in list(data['Equipment'].values())]
    columns = flatten(equipment_signals)
    return data,columns

def flatten(nested_list):
    flattened = []

    def _flatten(item):
        if isinstance(item, list):
            for sub_item in item:
                _flatten(sub_item)
        else:
            flattened.append(item)

    _flatten(nested_list)
    return flattened

def load_all_regions():
    base_path = Path(f"/data2/shreyas/bsp/V3/REGIONS")
    regions = list(base_path.glob('*.yaml'))
    infos, columns = [], []
    for region in regions:
        _i,_c = load_region(region)
        infos.extend(_i)
        columns.extend(_c)
    return infos, list(set(columns))


# def get_data(profile, region, limit_to_known=False):

#     if limit_to_known:
#         cobble_events = pd.read_csv('../DATA_ANALYSIS/cobble_events.csv')
#         cobble_events['start'] = pd.to_datetime(cobble_events['start'])
#         cobble_events['anomaly_start_time'] = cobble_events['start'] - pd.Timedelta(minutes=7)
#         cobble_events['anomaly_end_time'] = cobble_events['start'] - pd.Timedelta(seconds=1)
#         cobble_profile = cobble_events[cobble_events['profile']==profile].reset_index(drop=True)

#     if Path(f'../DATA/preprocessed_{profile}.pkl').exists():
#         with open(f'../DATA/preprocessed_{profile}.pkl','rb') as f:
#             data = pickle.load(f)

#         df_mill_on = data['mill_on']
#         cobble_removed_df = data['cobble_removed']
#         filtered_df = data['training']
#         anomaly_df = data['anomaly']
#         cobble_intervals = data['cobble_intervals']

#     else:
#         df_profile = load_data(profile)
#         cobble_events = None
#         if limit_to_known:
#             start_times = cobble_profile['start'].to_list()
#             df_profile = df_profile[(df_profile.index >= pd.Timestamp(start_times[0].date())) & (df_profile.index <= (pd.Timestamp(start_times[-1].date()) + pd.Timedelta(days=1)))]
#         df_mill_on, cobble_removed_df, filtered_df, cobble_intervals, anomaly_df = preprocess(df_profile)
#         with open(f'../DATA/preprocessed_{profile}.pkl','wb') as f:
#             pickle.dump({
#                 'mill_on': df_mill_on,
#                 'cobble_removed': cobble_removed_df,
#                 'training': filtered_df,
#                 'cobble_intervals': cobble_intervals,
#                 'anomaly': anomaly_df
#             },f)

#     region_info, columns = load_region(region)

#     if limit_to_known:
#         cobble_intervals = cobble_events

#     return {
#         'cobble_removed': cobble_removed_df.loc[:,columns],
#         'training': filtered_df.loc[:,columns],
#         'columns': columns,
#         'region_info': region_info
#     }


def get_data(profile, region, limit_to_known=False):

    region_info,columns = load_region(region)

    base_path = Path(f"/data2/shreyas/bsp/DATA/iba_data_1s/iba_data_for_training/{profile}")

    files = list(sorted(base_path.glob('*.parquet')))

    df = []
    missed = []
    for f in tqdm(files):
        try:
            _df = pd.read_parquet(f)
            df.append(_df.loc[:,columns])
        except Exception as e:
            print(e)
            print('couldnt read file:',f)
            missed.append(f)
    df = pd.concat(df)
    df = df.sort_index()

    with open(base_path / 'preprocessed' / f'{profile}_autoencoder_train_data.pkl','rb') as f:
        index = pickle.load(f)
        clean_index = [i for i in index if i>=df.index[0] and i<=df.index[-1]]
    with open(base_path / 'preprocessed' / f'{profile}_hypertuning_data.pkl','rb') as f:
        index = pickle.load(f)
        index1,index2 = index['train_df'].tolist(),index['test_df'].tolist()
        index = index1+index2
        cobble_removed_index = [i for i in index if i>=df.index[0] and i<=df.index[-1]]

    df = df.sort_index()
    df = df.loc[index,~df.columns.duplicated()]

    cobble_removed = df.loc[cobble_removed_index,:]
    training = df.loc[clean_index,:]

    if limit_to_known:
        cobble_intervals = pd.read_csv('/data2/shreyas/bsp/KDD/DATA_ANALYSIS/cobble_events.csv')
        cobble_intervals = cobble_intervals[cobble_intervals['profile']==profile].reset_index()
    else:
        cobble_intervals = pd.read_excel(base_path / 'preprocessed' / f'{profile}_cobble_intervals.xlsx')

    return {
        'cobble_removed': cobble_removed,
        'training': training,
        'cobble_intervals': cobble_intervals,
        'region_info': region_info,
        'columns': columns
    }


if __name__ == '__main__':
    data = get_data(profile='16mm',region='FFB_L1')
    data = get_data(profile='12mm',region='FFB_L1')