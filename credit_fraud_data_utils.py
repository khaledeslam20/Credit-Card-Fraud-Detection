from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as imb_Pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
import joblib

def load_data(train_path, val_path=None, val_size=0.2, random_state=42):
    train_df = pd.read_csv(train_path)
    if val_path :
        val_df = pd.read_csv(val_path)
    else :
        train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['Class'] ,
                                            random_state=random_state)

    x_train = train_df.drop(columns=['Class'])
    y_train = train_df['Class']

    x_val = val_df.drop(columns=['Class'])
    y_val = val_df['Class']

    print("Original train class distribution:")
    print(y_train.value_counts().sort_index())
    print(Counter(y_train))

    return x_train, y_train, x_val, y_val

def load_test_data(test_path):
    test_df = pd.read_csv(test_path)
    x_test = test_df.drop(columns=['Class'])
    y_test = test_df['Class']
    return x_test, y_test


def get_scaler( scaler=None):
    if scaler.lower() == "standard":
        return StandardScaler()
    elif scaler.lower() == "minmax":
        return MinMaxScaler()
    elif scaler.lower() == "robust":
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler}")




def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def apply_scaling(x_train, x_val, x_test=None, scaler="standard"): # need modification
    scaler = get_scaler(scaler)
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test) if x_test is not None else None
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler

def Counter(y) :
    cnt = y.value_counts().sort_index()
    return cnt




def get_sampler(method, y_train, under_ratio=None, over_ratio=None):
    if method == "undersample":
        minority_size = y_train.value_counts()[1]
        majority_size = int(minority_size * (under_ratio if under_ratio else 2))
        return RandomUnderSampler(sampling_strategy={0: majority_size}, random_state=42)
    elif method == "oversample":
        majority_size = y_train.value_counts()[0]
        new_size = int(majority_size / (over_ratio if over_ratio else 2))
        return SMOTE(sampling_strategy={1: new_size}, k_neighbors=4, random_state=42)
    elif method == "over_and_under":
        half_size = int(y_train.value_counts()[0] // 2)
        oversample = SMOTE(sampling_strategy={1: half_size}, k_neighbors=4, random_state=42)
        undersample = RandomUnderSampler(sampling_strategy={0: half_size}, random_state=42)
        return imb_Pipeline(steps=[('over', oversample), ('under', undersample)])
    else:
        return None
def apply_sampling (x_train, y_train, method=None,under_ratio=None, over_ratio=None) :
    sampler = get_sampler(method, y_train, under_ratio, over_ratio)
    if sampler is None:
        return x_train, y_train


    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    print(Counter(y_resampled))
    print(f"Train class distribution after {method}:")
    print(pd.Series(y_resampled).value_counts().sort_index())
    return x_resampled, y_resampled





# def underSampling(y_train) :
#
#     fac, minority_size = 2, Counter(y_train)[1]
#     sampler = RandomUnderSampler(sampling_strategy = {0:fac * minority_size}, random_state = 42)
#
#     return sampler
#
#
# def OverSampling(y_train):
#     from imblearn.over_sampling import SMOTE
#     fac, majoirty_size = 2, Counter(y_train)[0]  # best fact is two when i change it the f1_score decrease
#
#     new_size = int(majoirty_size / fac)
#
#     sampler = SMOTE(sampling_strategy={1: new_size}, k_neighbors=4, random_state=42)
#     # sampler = RandomOverSampler( random_state=42)
#     return sampler
#
#
# def Undersampling_oversampling(y_train):
#
#     minority_size = int(Counter(y_train)[0] // 2)
#     majority_size = int(Counter(y_train)[0] // 2)
#
#     oversample = SMOTE(sampling_strategy={1: minority_size}, k_neighbors=4, random_state=42)
#     undersample = RandomUnderSampler(sampling_strategy={0: majority_size}, random_state=42)
#
#     pip = imb_Pipeline(steps=[('over', oversample), ('under', undersample)])
#
#     return  pip