# Copyright (c) 2023 linjing-lab

from ._typing import Tuple, Optional
from ._utils import BaseModel, torch

class Regressier(BaseModel):
    '''
    Supervised Learning Regressier for Tabular Data.
    :param input_: int, input dataset with features' dimension of tabular data is input_.
    :param hidden_layer_sizes: Tuple[int], configure the size of each hidden layer. default: (100,).
    :param activation: str, configure function that activates the hidden layer. default: relu.
    :param criterion: str, loss function determined by different learning problem. default: MSELoss.
    :param solver: str, optimization function initialized with `learning_rate_init`. default: adam.
    :param batch_size: int, batch size of dataset in one training and validation process. default: 32.
    :param learning_rate_init: float, initialize the learning rate of the optimizer. default: 1e-2.
    :param lr_scheduler: str | None, set the learning rate scheduler integrated with optimizer. default: None.
    '''
    def __init__(self, 
                 input_: int, 
                 hidden_layer_sizes: Tuple[int]=(100,),
                 *,
                 activation: str='relu', 
                 criterion: str='MSELoss',
                 solver: str='adam', 
                 batch_size: int=32, 
                 learning_rate_init: float=1e-2,
                 lr_scheduler: Optional[str]=None) -> None:

        super(Regressier, self).__init__(input_, 
                                         1, 
                                         hidden_layer_sizes, 
                                         torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                         self._activate(activation), 
                                         self._criterion(criterion), 
                                         solver, 
                                         batch_size, 
                                         learning_rate_init, 
                                         lr_scheduler)

    def _activate(self, activation: str):
        '''
        Configure Activation with `activation` and `inplace=True`.
        :param activation: str, 'relu', 'rrelu', 'leaky_relu', 'elu', 'celu'. default: 'relu'.
        '''
        if activation == 'relu':
            return torch.nn.ReLU(inplace=True)
        elif activation == 'rrelu':
            return torch.nn.RReLU(inplace=True)
        elif activation == 'leaky_relu':
            return torch.nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            return torch.nn.ELU(inplace=True)
        elif activation == 'celu':
            return torch.nn.CELU(inplace=True)
        else:
            raise ValueError("Activation Function Supports Options: relu, rrelu, leaky_relu, elu, celu.")

    def _criterion(self, criterion: str):
        '''
        Configure Loss Criterion with `criterion`.
        :param criterion: str, 'MSELoss', 'L1Loss', 'SmoothL1Loss', 'KLDivLoss'. default: MSELoss.
        '''
        if criterion == 'MSELoss':
            return torch.nn.MSELoss()
        elif criterion == 'L1Loss':
            return torch.nn.L1Loss()
        elif criterion == 'SmoothL1Loss':
            return torch.nn.SmoothL1Loss()
        elif criterion == 'KLDivLoss':
            return torch.nn.KLDivLoss()
        else:
            raise ValueError("Criterion Configuration Supports Options: MSELoss, L1Loss, SmoothL1Loss, KLDivLoss.")

class Binarier(BaseModel):
    '''
    Binary Supervised Learning Classifier for Tabular Data.
    :param input_: int, input dataset with features' dimension of tabular data is input_.
    :param hidden_layer_sizes: Tuple[int], configure the size of each hidden layer. default: (100,).
    :param activation: str, configure function that activates the hidden layer. default: relu.
    :param criterion: str, loss function determined by different learning problem. default: CrossEntropyLoss.
    :param solver: str, optimization function coordinated with `torch.optim.lr_scheduler`. default: adam.
    :param batch_size: int, batch size of dataset in one training and validation process. default: 32.
    :param learning_rate_init: float, initialize the learning rate of the optimizer. default: 1e-2.
    :param lr_scheduler: str | None, set the learning rate scheduler integrated with optimizer. default: None.
    '''
    def __init__(self, 
                 input_: int, 
                 hidden_layer_sizes: Tuple[int]=(100,),
                 *,
                 activation: str='relu', 
                 criterion: str='CrossEntropyLoss',
                 solver: str='adam', 
                 batch_size: int=32, 
                 learning_rate_init: float=1e-2,
                 lr_scheduler: Optional[str]=None) -> None:

        super(Binarier, self).__init__(input_, 
                                       2, 
                                       hidden_layer_sizes, 
                                       torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                       self._activate(activation), 
                                       self._criterion(criterion), 
                                       solver, 
                                       batch_size, 
                                       learning_rate_init, 
                                       lr_scheduler)

    def _activate(self, activation: str):
        '''
        Configure Activation with `activation` and partly `inplace=True`.
        :param activation: str, 'relu', 'tanh', 'sigmoid', 'rrelu', 'leaky_relu', 'elu', 'celu'.
        '''
        if activation == 'relu':
            return torch.nn.ReLU(inplace=True)
        elif activation == 'tanh':
            return torch.nn.Tanh()
        elif activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation == 'rrelu':
            return torch.nn.RReLU(inplace=True)
        elif activation == 'leaky_relu':
            return torch.nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            return torch.nn.ELU(inplace=True)
        elif activation == 'celu':
            return torch.nn.CELU(inplace=True)
        else:
            raise ValueError("Activation Function Supports Options: relu, tanh, sigmoid, rrelu, leaky_relu, elu, celu.")

    def _criterion(self, criterion: str):
        '''
        Configure Loss Criterion with `criterion`.
        :param criterion: str, 'CrossEntropyLoss', 'BCEWithLogitsLoss'. default: CrossEntropyLoss.
        '''
        if criterion == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss()
        elif criterion == 'BCEWithLogitsLoss': # ! target and input need to be same size when adopt
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Criterion Configuration Supports Options: CrossEntropyLoss, BCEWithLogitsLoss.")

class Mutipler(BaseModel):
    '''
    Mutiple Supervised Learning Classifier for Tabular Data.
    :param input_: int, input dataset with features' dimension of tabular data is input_.
    :param num_classes: int, total number of correct label categories.
    :param hidden_layer_sizes: Tuple[int], configure the size of each hidden layer. default: (100,).
    :param activation: str, configure function that activates the hidden layer. default: relu.
    :param criterion: str, loss function determined by different learning problem. default: CrossEntropyLoss.
    :param solver: str, optimization function coordinated with `torch.optim.lr_scheduler`. default: adam.
    :param batch_size: int, batch size of dataset in one training and validation process. default: 32.
    :param learning_rate_init: float, initialize the learning rate of the optimizer. default: 1e-2.
    :param lr_scheduler: str | None, set the learning rate scheduler integrated with optimizer. default: None.
    '''
    def __init__(self, 
                 input_: int, 
                 num_classes: int,
                 hidden_layer_sizes: Tuple[int]=(100,),
                 *,
                 activation: str='relu', 
                 criterion: str='CrossEntropyLoss',
                 solver: str='adam', 
                 batch_size: int=32, 
                 learning_rate_init: float=1e-2,
                 lr_scheduler: Optional[str]=None) -> None:

        super(Mutipler, self).__init__(input_, 
                                       num_classes, 
                                       hidden_layer_sizes, 
                                       torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                       self._activate(activation), 
                                       self._criterion(criterion), 
                                       solver, 
                                       batch_size, 
                                       learning_rate_init, 
                                       lr_scheduler)
        assert num_classes >= 2

    def _activate(self, activation: str):
        '''
        Configure Activation with `activation` and `inplace=True`.
        :param activation: str, 'relu', 'rrelu', 'leaky_relu', 'elu', 'celu'.
        '''
        if activation == 'relu':
            return torch.nn.ReLU(inplace=True)
        elif activation == 'rrelu':
            return torch.nn.RReLU(inplace=True)
        elif activation == 'leaky_relu':
            return torch.nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            return torch.nn.ELU(inplace=True)
        elif activation == 'celu':
            return torch.nn.CELU(inplace=True)
        else:
            raise ValueError("Activation Function Supports Options: relu, rrelu, leaky_relu, elu, celu.")

    def _criterion(self, criterion: str):
        '''
        Configure Loss Criterion with `criterion`.
        :param criterion: str, 'CrossEntropyLoss', 'NLLLoss'.
        '''
        if criterion == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss()
        elif criterion == 'NLLLoss':
            return torch.nn.NLLLoss()
        else:
            raise ValueError("Criterion Configuration Supports Options: CrossEntropyLoss, NLLLoss.")
        
class Ranker(BaseModel):
    '''
    Supervised Learning Outputs Ranker for Tabular Data.
    :param input_: int, input dataset with features' dimension of tabular data is input_.
    :param num_outputs: int, total number of correct label outputs.
    :param hidden_layer_sizes: Tuple[int], configure the size of each hidden layer. default: (100,).
    :param activation: str, configure function that activates the hidden layer. default: relu.
    :param criterion: str, loss function determined by different learning problem. default: MultiLabelSoftMarginLoss.
    :param solver: str, optimization function coordinated with `torch.optim.lr_scheduler`. default: adam.
    :param batch_size: int, batch size of dataset in one training and validation process. default: 32.
    :param learning_rate_init: float, initialize the learning rate of the optimizer. default: 1e-2.
    :param lr_scheduler: str | None, set the learning rate scheduler integrated with optimizer. default: None.
    '''
    def __init__(self, 
                 input_: int, 
                 num_outputs: int,
                 hidden_layer_sizes: Tuple[int]=(100,),
                 *,
                 activation: str='relu', 
                 criterion: str='MultiLabelSoftMarginLoss',
                 solver: str='adam', 
                 batch_size: int=32, 
                 learning_rate_init: float=1e-2,
                 lr_scheduler: Optional[str]=None) -> None:

        super(Ranker, self).__init__(input_, 
                                     num_outputs, 
                                     hidden_layer_sizes, 
                                     torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                     self._activate(activation), 
                                     self._criterion(criterion),
                                     solver, 
                                     batch_size, 
                                     learning_rate_init, 
                                     lr_scheduler)

    def _activate(self, activation: str):
        '''
        Configure Activation with `activation` and `inplace=True`.
        :param activation: str, 'relu', 'rrelu', 'leaky_relu', 'elu', 'celu'.
        '''
        if activation == 'relu':
            return torch.nn.ReLU(inplace=True)
        elif activation == 'rrelu':
            return torch.nn.RReLU(inplace=True)
        elif activation == 'leaky_relu':
            return torch.nn.LeakyReLU(inplace=True)
        elif activation == 'elu':
            return torch.nn.ELU(inplace=True)
        elif activation == 'celu':
            return torch.nn.CELU(inplace=True)
        else:
            raise ValueError("Activation Function Supports Options: relu, rrelu, leaky_relu, elu, celu.")

    def _criterion(self, criterion: str):
        '''
        Configure Loss Criterion with `criterion`.
        :param criterion: str, 'MultiLabelMarginLoss', 'BCEWithLogitsLoss', 'MSELoss'. default: MultiLabelMarginLoss
        '''
        if criterion == 'MultiLabelSoftMarginLoss': # torch.long
            return torch.nn.MultiLabelSoftMarginLoss()
        elif criterion == 'BCEWithLogitsLoss': # torch.float
            return torch.nn.BCEWithLogitsLoss()
        elif criterion == 'MSELoss': # torch.float
            return torch.nn.MSELoss()
        else:
            raise ValueError("Criterion Configuration Supports Options: MultiLabelSoftMarginLoss, BCEWithLogitsLoss, MSELoss.")