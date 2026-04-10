# Copyright (c) CAIRI AI Lab. All rights reserved

from .convlstm import ConvLSTM
from .e3dlstm import E3DLSTM
from .mau import MAU
from .mim import MIM
from .phydnet import PhyDNet
from .predrnn import PredRNN
from .predrnnpp import PredRNNpp
from .predrnnv2 import PredRNNv2
from .simvp import SimVP
from .tau import TAU
from .mmvp import MMVP
from .swinlstm import SwinLSTM_D, SwinLSTM_B
from .wast import WaST

from .timesnet import TimesNet
from .itransformer import iTransformer
from .timemixer import TimeMixer
from .timesformer import TimeSformer
from .dmvfn import DMVFN
from .PredFormer import PredFormer
from .yield3dcnn import Yield3DCNN
from .VMRNN import VMRNN_D, VMRNN_B
from .GMG import GMG

method_maps = {
    'convlstm': ConvLSTM,
    'e3dlstm': E3DLSTM,
    'mau': MAU,
    'mim': MIM,
    'phydnet': PhyDNet,
    'predrnn': PredRNN,
    'predrnnpp': PredRNNpp,
    'predrnnv2': PredRNNv2,
    'simvp': SimVP,
    'tau': TAU,
    'mmvp': MMVP,
    'swinlstm_d': SwinLSTM_D,
    'swinlstm_b': SwinLSTM_B,
    'swinlstm': SwinLSTM_B,
    'wast': WaST,
    'timesnet': TimesNet,
    'itransformer': iTransformer,
    'timemixer': TimeMixer,
    'timesformer': TimeSformer,
    'dmvfn': DMVFN,
    'yield3dcnn': Yield3DCNN,
    'predformer': PredFormer,
    'vmrnn_d': VMRNN_D,
    'vmrnn_b': VMRNN_B,
    'vmrnn': VMRNN_B,
    'gmg': GMG,
}

__all__ = [
    'method_maps', 'ConvLSTM', 'E3DLSTM', 'MAU', 'MIM',
    'PredRNN', 'PredRNNpp', 'PredRNNv2', 'PhyDNet', 'SimVP', 'TAU',
    "MMVP", 'SwinLSTM_D', 'SwinLSTM_B', 'WaST', 'TimesNet', 'iTransformer', 'TimeMixer', 'TimeSformer', 'DMVFN',
    'PredFormer', 'Yield3DCNN', 'VMRNN_D', 'VMRNN_B', 'GMG'
]
