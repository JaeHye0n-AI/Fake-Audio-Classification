from tqdm import tqdm
import IPython
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

import librosa
import pandas as pd
import glob
import shutil
import warnings
import random
from utils import *
import librosa
import pickle
import os
from librosa import feature



def mk_train_val_dataset(train_path, batch_size):
    feature_kind = "3FEATURE"

    feature_df = 0
    if feature_kind == "MFCC":

        if not os.path.exists(f"{train_path}/mfcc_feature.pkl"):
            print('preparing mfcc_train_feature')

            pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

            warnings.filterwarnings('ignore')
            data = []
            labels = []
            folders = ['fake', 'real']  # fake: 0, real: 1

            for folder in folders:
                file_paths = glob.glob(f"{train_path}/{folder}/*.wav")
                for curr_path in tqdm(file_paths):
                    audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
                    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                    padded_mfcc = pad2d(mfccs_features, 200)
                    data.append(padded_mfcc)
                    labels.append(folder)

            feature_df = pd.DataFrame({"features": data, "class": labels})

            feature_df["class"] = label_encoder(feature_df["class"])  # fake: 0, real: 1

            feature_df.to_pickle(f"{train_path}/mfcc_feature.pkl")

        else:
            print('already mfcc_train_feature')
            feature_df = pd.read_pickle(f"{train_path}/mfcc_feature.pkl")

    elif feature_kind == "SPECTOGRAM":

        if not os.path.exists(f"{train_path}/spectogram_feature.pkl"):
            print('preparing spectogram_train_feature')

            pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

            warnings.filterwarnings('ignore')
            data = []
            labels = []
            folders = ['fake', 'real']  # fake: 0, real: 1

            for folder in folders:
                file_paths = glob.glob(f"{train_path}/{folder}/*.wav")
                for curr_path in tqdm(file_paths):
                    audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
                    real_spec = np.abs(librosa.stft(audio))
                    real_spec = librosa.amplitude_to_db(real_spec, ref=np.max)
                    padded_spec = pad2d(real_spec, 100)
                    data.append(padded_spec)
                    labels.append(folder)

            feature_df = pd.DataFrame({"features": data, "class": labels})

            feature_df["class"] = label_encoder(feature_df["class"])  # fake: 0, real: 1

            feature_df.to_pickle(f"{train_path}/spectogram_feature.pkl")

        else:
            print('already spectogram_train_feature')
            feature_df = pd.read_pickle(f"{train_path}/spectogram_feature.pkl")

    elif feature_kind == "MEL_SPECTOGRAM":

        if not os.path.exists(f"{train_path}/mel_spectogram_feature.pkl"):
            print('preparing mel_spectogram_train_feature')

            pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

            warnings.filterwarnings('ignore')
            data = []
            labels = []
            folders = ['fake', 'real']  # fake: 0, real: 1

            for folder in folders:
                file_paths = glob.glob(f"{train_path}/{folder}/*.wav")
                for curr_path in tqdm(file_paths):
                    audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
                    real_mel_spec = feature.melspectrogram(y=audio, sr=sample_rate)
                    real_mel_spec = librosa.power_to_db(real_mel_spec, ref=np.max)
                    padded_mel_spec = pad2d(real_mel_spec, 100)
                    data.append(padded_mel_spec)
                    labels.append(folder)


            feature_df = pd.DataFrame({"features": data, "class": labels})

            feature_df["class"] = label_encoder(feature_df["class"])  # fake: 0, real: 1

            feature_df.to_pickle(f"{train_path}/mel_spectogram_feature.pkl")

        else:
            print('already mel_spectogram_train_feature')
            feature_df = pd.read_pickle(f"{train_path}/mel_spectogram_feature.pkl")

    elif feature_kind == "CHROMAGRAM":

        if not os.path.exists(f"{train_path}/chromagram_feature.pkl"):
            print('preparing chromagram_train_feature')

            pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

            warnings.filterwarnings('ignore')
            data = []
            labels = []
            folders = ['fake', 'real']  # fake: 0, real: 1

            for folder in folders:
                file_paths = glob.glob(f"{train_path}/{folder}/*.wav")
                for curr_path in tqdm(file_paths):
                    audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
                    real_chroma = feature.chroma_cqt(y=audio, sr=sample_rate, bins_per_octave=36)
                    padded_chroma = pad2d(real_chroma, 100)
                    data.append(padded_chroma)
                    labels.append(folder)

            feature_df = pd.DataFrame({"features": data, "class": labels})

            feature_df["class"] = label_encoder(feature_df["class"])  # fake: 0, real: 1

            feature_df.to_pickle(f"{train_path}/chromagram_feature.pkl")

        else:
            print('already chromagram_train_feature')
            feature_df = pd.read_pickle(f"{train_path}/chromagram_feature.pkl")

    elif feature_kind == "3FEATURE":
        with open(f"{train_path}/spectogram_feature.pkl", "rb") as fh:
            feature_df_1 = pickle.load(fh)

        with open(f"{train_path}/mel_spectogram_feature.pkl", "rb") as fh:
            feature_df_2 = pickle.load(fh)

        with open(f"{train_path}/chromagram_feature.pkl", "rb") as fh:
            feature_df_3 = pickle.load(fh)

        X_1 = np.asarray(feature_df_1["features"].tolist())
        X_2 = np.asarray(feature_df_2["features"].tolist())
        X_3 = np.asarray(feature_df_3["features"].tolist())

        X = np.concatenate((X_1, X_2, X_3), axis=1)

        y = np.asarray(feature_df_1["class"].tolist())

        # 3feature dim ==> x1=1025, x2=128, x3=12


    ### when use multi feature, under two line change to dummy ###########
    # X = np.asarray(feature_df["features"].tolist())
    # y = np.asarray(feature_df["class"].tolist())

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # First step: converting to tensor
    x_train_to_tensor = torch.from_numpy(X_train).to(torch.float32)
    y_train_to_tensor = torch.from_numpy(y_train).to(torch.long)
    x_val_to_tensor = torch.from_numpy(X_val).to(torch.float32)
    y_val_to_tensor = torch.from_numpy(y_val).to(torch.long)

    # Second step: Creating TensorDataset for Dataloader
    train_dataset = TensorDataset(x_train_to_tensor, y_train_to_tensor)
    val_dataset = TensorDataset(x_val_to_tensor, y_val_to_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def mk_test_dataset(test_path, batch_size=32):
    feature_kind = "3FEATURE"

    if feature_kind == "MFCC":

        if not os.path.exists(f"{test_path}/mfcc_feature.pkl"):
            print('preparing mfcc_test_feature')

            pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

            test_data = []
            test_file_paths = sorted(glob.glob(f"{test_path}/*.wav"))
            for curr_path in tqdm(test_file_paths):
                audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
                mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                padded_mfcc = pad2d(mfccs_features, 200)
                test_data.append(padded_mfcc)
            with open(f"{test_path}/mfcc_feature.pkl", "wb") as f:
                pickle.dump(test_data, f)

        else:
            print('already mfcc_test_feature')
            with open(f"{test_path}/mfcc_feature.pkl", "rb") as f:
                test_data = pickle.load(f)

    elif feature_kind == "SPECTOGRAM":

        if not os.path.exists(f"{test_path}/spectogram_feature.pkl"):
            print('preparing spectogram_test_feature')

            pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

            test_data = []
            test_file_paths = sorted(glob.glob(f"{test_path}/*.wav"))
            for curr_path in tqdm(test_file_paths):
                audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
                real_spec = np.abs(librosa.stft(audio))
                real_spec = librosa.amplitude_to_db(real_spec, ref=np.max)
                padded_spec = pad2d(real_spec, 100)
                test_data.append(padded_spec)
            with open(f"{test_path}/spectogram_feature.pkl", "wb") as f:
                pickle.dump(test_data, f)

        else:
            print('already spectogram_test_feature')
            with open(f"{test_path}/spectogram_feature.pkl", "rb") as f:
                test_data = pickle.load(f)

    elif feature_kind == "MEL_SPECTOGRAM":

        if not os.path.exists(f"{test_path}/mel_spectogram_feature.pkl"):
            print('preparing mel_spectogram_test_feature')

            pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

            test_data = []
            test_file_paths = sorted(glob.glob(f"{test_path}/*.wav"))
            for curr_path in tqdm(test_file_paths):
                audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
                real_mel_spec = feature.melspectrogram(y=audio, sr=sample_rate)
                real_mel_spec = librosa.power_to_db(real_mel_spec, ref=np.max)
                padded_mel_spec = pad2d(real_mel_spec, 100)
                test_data.append(padded_mel_spec)
            with open(f"{test_path}/mel_spectogram_feature.pkl", "wb") as f:
                pickle.dump(test_data, f)

        else:
            print('already mel_spectogram_test_feature')
            with open(f"{test_path}/mel_spectogram_feature.pkl", "rb") as f:
                test_data = pickle.load(f)

    elif feature_kind == "CHROMAGRAM":

        if not os.path.exists(f"{test_path}/chromagram_feature.pkl"):
            print('preparing chromagram_test_feature')

            pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

            test_data = []
            test_file_paths = sorted(glob.glob(f"{test_path}/*.wav"))
            for curr_path in tqdm(test_file_paths):
                audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
                real_chroma = feature.chroma_cqt(y=audio, sr=sample_rate, bins_per_octave=36)
                padded_chroma = pad2d(real_chroma, 100)
                test_data.append(padded_chroma)
            with open(f"{test_path}/chromagram_feature.pkl", "wb") as f:
                pickle.dump(test_data, f)

        else:
            print('already chromagram_test_feature')
            with open(f"{test_path}/chromagram_feature.pkl", "rb") as f:
                test_data = pickle.load(f)

    elif feature_kind == "3FEATURE":
        with open(f"{test_path}/spectogram_feature.pkl", "rb") as f:
            test_data_1 = pickle.load(f)

        with open(f"{test_path}/mel_spectogram_feature.pkl", "rb") as f:
            test_data_2 = pickle.load(f)

        with open(f"{test_path}/chromagram_feature.pkl", "rb") as f:
            test_data_3 = pickle.load(f)

        X_test_1 = np.asarray(test_data_1)
        X_test_2 = np.asarray(test_data_2)
        X_test_3 = np.asarray(test_data_3)

        X_test = np.concatenate((X_test_1, X_test_2, X_test_3), axis=1)


    # X_test = np.asarray(test_data)
    x_test_to_tensor = torch.from_numpy(X_test).to(torch.float32)
    test_dataset = TensorDataset(x_test_to_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    return test_loader