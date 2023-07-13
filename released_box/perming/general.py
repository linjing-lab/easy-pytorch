# Copyright (c) 2023 linjing-lab

from ._typing import Tuple, Optional
from ._utils import BaseModel, torch

class Box(BaseModel):
    '''
    Supervised Learning Box for Tabular Data.
    :param input_: int, features' dimension of tabular data is input_.
    :param num_classes: int, total number of correct label categories.
    :param hidden_layer_sizes: Tuple[int], configure the size of each hidden layer. default: (100,).
    :param device: str, configure hardware environment for training. options: "cuda", "cpu". default: cuda.
    :param activation: str, configure function that activates the hidden layer. default: relu.
    :param inplace_on: bool, whether to use `inplace=True` on activation. default: False.
    :param criterion: str, loss function determined by different learning problem. default: CrossEntropyLoss.
    :param solver: str, optimization coordinated with `torch.optim.lr_scheduler`. default: adam.
    :param batch_size: int, batch size of tabular dataset in one training process. default: 32.
    :param learning_rate_init: float, initialize the learning rate of the optimizer. default: 1e-2.
    :param lr_scheduler: str | None, set the learning rate scheduler integrated with the optimizer. default: None.
    '''
    def __init__(self, 
                 input_: int, 
                 num_classes: int, # num_classes=1
                 hidden_layer_sizes: Tuple[int]=(100,),
                 device: str='cuda',
                 *,
                 activation: str='relu', 
                 inplace_on: bool=False, 
                 criterion: str='CrossEntropyLoss', # criterion='MSELoss'
                 solver: str='adam', 
                 batch_size: int=32, 
                 learning_rate_init: float=1e-2,
                 lr_scheduler: Optional[str]=None) -> None:
        
        super(Box, self).__init__(input_, 
                                  num_classes, 
                                  hidden_layer_sizes, 
                                  self._device(device), 
                                  self._activate(activation, inplace_on), 
                                  self._criterion(criterion), 
                                  solver, 
                                  batch_size, 
                                  learning_rate_init, 
                                  lr_scheduler)

    def _device(self, option: str) -> None:
        '''
        Set Device for Model and Training.
        :param option: device configuration. str, 'cuda' or 'cpu'.
        '''
        if option == 'cuda':
            return torch.device(option if torch.cuda.is_available() else 'cpu')
        elif option == 'cpu':
            return torch.device(option)
        else:
            raise ValueError("Box Only Support 2 Options for Device Configuration: 'cuda' or 'cpu'.")

    def _activate(self, activation: str, inplace_on: bool):
        '''
        Configure Activation with `activation` and `inplace_on`.
        :param activation: str, 'relu', 'tanh', 'sigmoid', 'rrelu', 'leaky_relu', 'prelu', 'softplus', 'elu', 'celu'.
        :param inplace_on: bool, whether to use `inplace=True` on activations. default: False.
        '''
        if activation == 'relu': # most use
            return torch.nn.ReLU(inplace=inplace_on)
        elif activation == 'tanh':
            return torch.nn.Tanh()
        elif activation == 'sigmoid': # can be used in classification = 2
            return torch.nn.Sigmoid()
        elif activation == 'rrelu':
            return torch.nn.RReLU(inplace=inplace_on)
        elif activation == 'leaky_relu':
            return torch.nn.LeakyReLU(inplace=inplace_on)
        elif activation == 'prelu':
            return torch.nn.PReLU()
        elif activation == 'softplus':
            return torch.nn.Softplus()
        elif activation == 'elu':
            return torch.nn.ELU(inplace=inplace_on)
        elif activation == 'celu':
            return torch.nn.CELU(inplace=inplace_on)
        else:
            raise ValueError("Activation Function Supports Options: relu, tanh, sigmoid, rrelu, leaky_relu, softplus, elu, celu.")

    def _criterion(self, criterion: str):
        '''
        Configure Loss Criterion with `criterion`.
        :param criterion: str, 'CrossEntropyLoss', 'NLLLoss', 'MultiLabelMarginLoss', 'BCELoss', 'BCEWithLogitsLoss', 'MSELoss', 'L1Loss', 'SmoothL1Loss', 'KLDivLoss'. default: CrossEntropyLoss.
        '''
        if criterion == 'CrossEntropyLoss': # classification with num_classes > 2.
            return torch.nn.CrossEntropyLoss()
        elif criterion == 'NLLLoss':
            return torch.nn.NLLLoss()
        elif criterion == 'MultiLabelSoftMarginLoss':
            return torch.nn.MultiLabelSoftMarginLoss()
        elif criterion == 'BCELoss': # classification with num_classes = 2
            return torch.nn.BCELoss()
        elif criterion == 'BCEWithLogitsLoss':
            return torch.nn.BCEWithLogitsLoss()
        elif criterion == 'MSELoss': # regression
            return torch.nn.MSELoss()
        elif criterion == 'L1Loss':
            return torch.nn.L1Loss()
        elif criterion == 'SmoothL1Loss':
            return torch.nn.SmoothL1Loss()
        elif criterion == 'KLDivLoss':
            return torch.nn.KLDivLoss()
        else:
            raise ValueError("Criterion Configuration Supports Options: CrossEntropyLoss, NLLLoss, MultiLabelSoftMarginLoss, BCELoss, BCEWithLogitsLoss, MSELoss, L1Loss, SmoothL1Loss, KLDivLoss.") 