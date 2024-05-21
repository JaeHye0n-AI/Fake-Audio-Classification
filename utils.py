import IPython
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import numpy as np



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def label_encoder(column):
    le = LabelEncoder().fit(column)
    print(column.name, le.classes_)
    return le.transform(column)

def save_config(config, file_path):
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(str(config))
    fo.close()