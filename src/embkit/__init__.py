"""
Embedding Kit

The base module contains frequently used methods and functions
"""
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    get_device - scan system and return default device
    """
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.info("Using Metal GPU")
        return torch.device("mps")
    logger.info("Using CPU")
    return torch.device("cpu")

def dataframe_loader(df: pd.DataFrame,
                     batch_size = 256, shuffle=True,
                     device=None) -> torch.utils.data.DataLoader:
    """
    dataframe_loader

    :param df: Pandas Dataframe to be converted into passed to DataLoader
    :param batch_size: value to be passed to DataLoader batch_size
    :param shuffle: value to be passed to DataLoader shuffle
    :param device: Device the where tensor should be moved. If None, use get_device 
    """

    if device is None:
        device = get_device()
    x = torch.from_numpy(df.values.astype(np.float32)).to(device)
    dataset = TensorDataset(x)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
