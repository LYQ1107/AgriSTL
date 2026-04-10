# Copyright (c) CAIRI AI Lab. All rights reserved

from .dataloader import load_data
from .dataset_constant import dataset_parameters
from .pipelines import *
from .utils import create_loader
from .base_data import BaseDataModule

try:
    from .dataloader_human import HumanDataset
except Exception:
    HumanDataset = None

try:
    from .dataloader_kitticaltech import KittiCaltechDataset
except Exception:
    KittiCaltechDataset = None

try:
    from .dataloader_kth import KTHDataset
except Exception:
    KTHDataset = None

try:
    from .dataloader_moving_mnist import MovingMNIST
except Exception:
    MovingMNIST = None

try:
    from .dataloader_namin import NaminDataset
except Exception:
    NaminDataset = None

try:
    from .dataloader_yield import YieldDataset
except Exception:
    YieldDataset = None

try:
    from .dataloader_taxibj import TaxibjDataset
except Exception:
    TaxibjDataset = None

try:
    from .dataloader_weather import WeatherBenchDataset
except Exception:
    WeatherBenchDataset = None

try:
    from .dataloader_sevir import SEVIRDataset
except Exception:
    SEVIRDataset = None

__all__ = [
    'KittiCaltechDataset', 'HumanDataset', 'KTHDataset', 'MovingMNIST', 'TaxibjDataset',
    'WeatherBenchDataset', 'SEVIRDataset',
    'load_data', 'dataset_parameters', 'create_loader', 'BaseDataModule', 'NaminDataset',
    'YieldDataset'
]
