import os
import logging
import requests
import scipy.io as sio
from torch.utils.data import Dataset
import torch

def load_dataset(name, dir='dataset'):
    print("read dataset:", name)
    if not os.path.exists(dir):
        logging.info(f'Directory {dir} does not exist, creating it.')
        os.makedirs(dir)
    dataset_path = os.path.join(dir, name+'.mat')
    print("path:", dataset_path)
    if not os.path.exists(dataset_path):
        logging.info(f'Dataset {name}.mat does not exist, downloading it now, please wait...')
        url = f'https://raw.githubusercontent.com/SpriteMisaka/PyLDL/main/dataset/{name}.mat'
        response = requests.get(url)
        if response.status_code == 200:
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            logging.info(f'Dataset {name}.mat downloaded successfully.')
        else:
            raise ValueError(f'Failed to download {name}.mat')
    data = sio.loadmat(dataset_path)
    print("successfully load.")
    print("features", data['features'].shape) # (1980, 168)
    print("labels", data['labels'].shape) # (1980, 7)
    return data['features'], data['labels']

class MatDataset(Dataset):
    def __init__(self, dataset_name, path):
        self.dataset_name = dataset_name
        self.X, self.y = load_dataset(dataset_name, dir=path)
        self.X = torch.Tensor(self.X)
        self.y = torch.Tensor(self.y)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]