# Copyright (c) 2023 linjing-lab

import torch
import random
import numpy
import gc
import sortingx
from joblib import parallel_backend
from collections import OrderedDict
from ._typing import TabularData, Tuple, Dict, Optional, Any
from ._version import parse_torch_version

class TabularDataset(torch.utils.data.Dataset):
    '''
    Tabular Data Constructed with `numpy.array` noted as `TabularData`.
    :param features: TabularData, composed of n-row samples and m-column features.
    :param target: TabularData, consists of correct labels or values with size at n.
    :param squeeze: bool, regression or classification. it represents a regression problem when `squeeze=True`.
    '''
    def __init__(self, features: TabularData, target: TabularData, squeeze: bool) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float)
        self.target = torch.as_tensor(target, dtype=torch.float) if squeeze else torch.as_tensor(target, dtype=torch.long)

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self) -> int:
        return len(self.features)

class MLP(torch.nn.Module):
    '''
    Construct Model Layers with `input_`, `num_classes`, `hidden_layer_sizes`, `activation`.
    :param input_: int, features' dimension of tabular data is input_.
    :param num_classes: int, total number of correct label categories.
    :param hidden_layer_sizes: Tuple[int], configure the size of each hidden layer. default: (100,).
    :param activation:, activation configured by Classfier. default: torch.nn.ReLU().
    '''
    def __init__(self, input_: int, num_classes: int, hidden_layer_sizes: Tuple[int], activation) -> None:
        super(MLP, self).__init__()
        self.squeeze = False if num_classes > 1 else True
        if hidden_layer_sizes:
            model_layers, hidden_length = OrderedDict(), len(hidden_layer_sizes)
            model_layers['Linear0'] = torch.nn.Linear(input_, hidden_layer_sizes[0])
            model_layers['Activation0'] = activation
            for index in range(1, hidden_length):
                linear_, activation_ = "Linear" + str(index), "Activation" + str(index)
                model_layers[linear_] = torch.nn.Linear(hidden_layer_sizes[index - 1], hidden_layer_sizes[index])
                model_layers[activation_] = activation
            model_layers["Linear" + str(hidden_length)] = torch.nn.Linear(hidden_layer_sizes[hidden_length - 1], num_classes)
            self.mlp = torch.nn.Sequential(model_layers)
        else:
            self.mlp = torch.nn.Linear(input_, num_classes)

    def forward(self, x):
        output = self.mlp(x)
        return output.squeeze(-1) if self.squeeze else output
    
class BaseModel:
    '''
    Basic Model for Configuring Common Models and General Box.
    :param input_: int, features' dimension of tabular data is input_.
    :param num_classes: int, total number of correct label categories.
    :param hidden_layer_sizes: Tuple[int], configure the size of each hidden layer.
    :param device: Any, device configured by common models and general box.
    :param activation: Any, activated function configured by common models and general box.
    :param criterion: Any, loss function determined by common models and general box.
    :param solver: str, optimization function coordinated with `torch.optim.lr_scheduler`. default: adam. (modified more params in `_solver`.)
    :param batch_size: int, batch size of dataset in one training process. default: 32.
    :param learning_rate_init: float, initialize the learning rate of the optimizer. default: 1e-3.
    :param lr_scheduler: str | None, set the learning rate scheduler integrated with the optimizer. default: None. (modifier more params in `_scheduler`.)
    '''
    def __init__(self,
                 input_: int,
                 num_classes: int,
                 hidden_layer_sizes: Tuple[int],
                 device: Any,
                 activation: Any,
                 criterion: Any,
                 solver: str, 
                 batch_size: int, 
                 learning_rate_init: float,
                 lr_scheduler: Optional[str]=None) -> None:

        assert input_ > 0
        assert num_classes > 0
        self.input: int = input_
        self.num_classes: int = num_classes
        self.activation = activation
        self.device = device # device configuration
        self.criterion = criterion
        self.batch_size: int = batch_size
        self.lr: float = learning_rate_init
        self.model = MLP(self.input, self.num_classes, hidden_layer_sizes, self.activation).to(self.device)
        if parse_torch_version(torch.__version__)[0] >= ('2', '0', '0'):
            self.model = torch.compile(self.model)
        self.solver = self._solver(solver)
        self.lr_scheduler = self._scheduler(lr_scheduler)

    def _solver(self, solver: str):
        '''
        Configure Optimizer with `solver`.
        :param solver: str, "sgd", "momentum", "adam", "adagrad", "rmsprop". default: adam.
        '''
        if solver == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif solver == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99))
        elif solver == 'momentum':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, nesterov=True)
        elif solver == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
        elif solver == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.9)
        else:
            raise ValueError("Optimizer Configuration Supports Options: sgd, momentum, adam, adagrad, rmsprop.")

    def _scheduler(self, lr_scheduler: Optional[str]):
        '''
        Configure Learning Rate Scheduler with `lr_scheduler`.
        :param lr_scheduler: str, "exponential_lr", "step_lr", "cosine_annealing_lr". default: None.
        '''
        if lr_scheduler != None:
            if lr_scheduler == 'exponential_lr':
                return torch.optim.lr_scheduler.ExponentialLR(self.solver, gamma=0.1)
            elif lr_scheduler == 'step_lr':
                return torch.optim.lr_scheduler.StepLR(self.solver, step_size=2, gamma=0.1)
            elif lr_scheduler == 'cosine_annealing_lr':
                return torch.optim.lr_scheduler.CosineAnnealingLR(self.solver, T_max=10, eta_min=0)
            else:
                raise ValueError("Learning Rate Scheduler Supports Options: exponential_lr, step_lr, cosine_annealing_lr.")

    def print_config(self):
        '''
        Print Model Sequential and Basic Configuration.
        '''
        print(self.model)
        return OrderedDict({
            'torch -v': torch.__version__,
            'criterion': self.criterion,
            'batch_size': self.batch_size,
            'solver': self.solver,
            'lr_scheduler': self.lr_scheduler,
            'device': self.device,
        })
    
    def data_loader(self, 
                    features: TabularData, 
                    target: TabularData, 
                    ratio_set: Dict[str, int]={'train': 8, 'test': 1, 'val': 1}, 
                    worker_set: Dict[str, int]={'train': 8, 'test': 2, 'val': 1},
                    random_seed: Optional[int]=None) -> None:
        '''
        Encapsulate Dataset to DataLoader from Scratch.
        :param features: TabularData, composed of n-row samples and m-column features.
        :param target: TabularData, consists of correct labels or values with size at n.
        :param ratio_set: Dict[str, int], stores the proportion of train-set, test-set and val-set. default: {'train': 7, 'test': 2, 'val': 1}.
        :param worker_set: Dict[str, int], load the configuration of DataLoader with multi-threaded. default: {'train': 8, 'test': 2, 'val': 1}.
        :param random_seed: int | None, random.seed(random_seed) used for fixed random datasets. default: None.
        '''
        assert ratio_set['train'] > 0 and ratio_set['test'] > 0 and ratio_set['val'] > 0
        assert ratio_set['train'] > 4 * ratio_set['test'], "The training set needs to be larger than the test set."
        assert ratio_set['train'] + ratio_set['test'] + ratio_set['val'] == 10, "The sum of 3 datasets' ratio needs to be 10."
        assert features.shape[1] == self.input, "Please ensure `input_` is equal to `features.shape[1]`."
        if self.num_classes >= 2:
            self.unique = numpy.unique(target)
            assert len(self.unique) == self.num_classes, "Please ensure `num_classes` is equal to `len(numpy.unique(labels))`."
            self.indices = dict(zip(self.unique, range(self.num_classes)))
            target = numpy.array([self.indices[value] for value in target], dtype=numpy.int8)
        train_, test_, val_ = train_test_val_split(features, target, ratio_set, random_seed)
        self.train_loader = torch.utils.data.DataLoader(TabularDataset(train_['features'], train_['target'], self.model.squeeze), batch_size=self.batch_size, shuffle=True, num_workers=worker_set['train'], )
        self.test_loader = torch.utils.data.DataLoader(TabularDataset(test_['features'], test_['target'], self.model.squeeze), batch_size=self.batch_size, shuffle=True, num_workers=worker_set['test'])
        self.val_loader = torch.utils.data.DataLoader(TabularDataset(val_['features'], val_['target'], self.model.squeeze), batch_size=self.batch_size, shuffle=False, num_workers=worker_set['val'])

    def train_val(self, 
                  num_epochs: int=2, 
                  interval: int=100, 
                  backend: str="threading", 
                  n_jobs: int=-1) -> None:
        '''
        Training and Validation with `train_loader` and `val_container`.
        :param num_epochs: int, training epochs for `self.model`. default: 5.
        :param interval: int, console output interval. default: 100.
        :param backend: str, "threading", "multiprocessing, 'locky'. default: "threading".
        :param n_jobs: int, accelerate processing of validation. default: -1.
        '''
        assert n_jobs == -1 or n_jobs > 0
        total_step = len(self.train_loader)
        self._set_container(backend, n_jobs)
        for epoch in range(num_epochs):
            gc.collect()
            torch.cuda.empty_cache()
            for i, (features, target) in enumerate(self.train_loader):
                features = features.to(self.device)
                target = target.to(self.device)

                # froward pass
                outputs = self.model(features)
                self.train_loss = self.criterion(outputs, target)

                # backward and optimize
                self.solver.zero_grad()
                self.train_loss.backward()
                _ = self.solver.step() if self.lr_scheduler == None else self.lr_scheduler.step()
                
                # validation with val_container
                self.val_loss = 0
                with parallel_backend(backend, n_jobs=n_jobs):
                    for val_set in self.val_container:
                        outputs_val = self.model(val_set[0].to(self.device))
                        self.val_loss += self.criterion(outputs_val, val_set[1].to(self.device))

                self.val_loss /= len(self.val_container)

                # console print
                if (i + 1) % interval == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, self.train_loss.item(), self.val_loss.item()))

    def test(self, 
             sort_by: str='accuracy', 
             sort_kernel: str='bubble', 
             sort_state: bool=True):
        '''
        Test with Initialized Configuration. `accuracy > 0`, 'correct_class != None' and the underline keywords only appears when `num_classes >= 2`.
        :param sort_by: str, 'accuracy', 'numbers', 'num-total'. default: 'accuracy'.
        :param sort_kernel: 'bubble', 'insert', 'shell', 'heap', 'quick', 'merge'. default: 'bubble'.
        :param sort_state: bool, whether to use descending order when sorting. default: True.
        '''
        with torch.no_grad():
            self.test_loss, correct, self.correct_class, test_loader_step = 0, 0, dict.fromkeys(self.unique, [0, 0]) if self.num_classes >= 2 else None, len(self.test_loader)
            test_total = test_loader_step * self.batch_size
            for features, target in self.test_loader:
                features = features.to(self.device)
                target = target.to(self.device)            
                outputs = self.model(features)
                if self.num_classes >= 2:
                    _, predicted = torch.max(outputs.data, 1)
                    for index, value in enumerate(predicted):
                        self.correct_class[self.unique[value]][1] += 1
                        self.correct_class[self.unique[value]][0] += (value == target[index]).item()
                    correct += (predicted == target).sum().item()
                self.test_loss += self.criterion(outputs, target)
            self.test_loss /= test_loader_step
            print('loss of {0} on the {1} test dataset: {2}. accuracy: {3:.4f} %'.format(self.__class__.__name__, test_total, self.test_loss.item(), 100 * correct / test_total))
        return OrderedDict(self._packing(sort_by, sort_kernel, sort_state))
    
    def save(self, show: bool=True, dir: str='./model') -> None:
        '''
        Save Model Checkpoint with Classifier.
        :param show: bool, whether to show `model.state_dict()`. default: True.
        :param dir: str, model save to. default: './model'.
        '''
        if show:
            print(self.model.state_dict())
        torch.save(self.model.state_dict(), dir)

    def load(self, show: bool=True, dir: str='./model') -> None:
        '''
        Load Model Checkpoint to Classifier.
        :param show: bool, whether to show `model.state_dict()`. default: True.
        :param dir: str, model load from. default: './model'.
        '''
        params = torch.load(dir)
        self.model.load_state_dict(params)
        if show:
            print(self.model.state_dict())

    def _packing(self, by: str, kernel: str, state: bool) -> Dict[str, Any]:
        '''
        Pack Test Returned Data including `correct_class` and returned result of `sortingx.[method]()`.
        :param by: str, choose which way to sort the order of 'correct_class'.
        :param kernel: str, choose which kernel used for sorting.
        :param state: bool, choose the state when `correct_class` is sorting.
        '''
        assert kernel in sortingx.__all__, "Please ensure kernel is sortingx.__all__."
        kernel = eval('sortingx.' + kernel)
        loss_, classify, regress = {
            'loss': {'train': self.train_loss.item(), 
                     'val': self.val_loss.item(),
                     'test': self.test_loss.item()}
        }, {'problem': 'classification',
            'num_classes': self.num_classes,
            'column': ('label name', ('true numbers', 'total numbers')),
            'labels': self.correct_class}, {'problem': 'regression'}
        if self.num_classes >= 2:
            classify.update(loss_)
            if by == 'numbers':
                classify.update({'sorted': kernel(self.correct_class.items(), lambda d: d[1][0], reverse=state)})
                return classify
            elif by == 'accuracy':
                classify.update({'sorted': kernel(self.correct_class.items(), lambda d: d[1][0]/d[1][1], reverse=state)})
                return classify
            elif by == 'num-total':
                classify.update({'sorted': kernel(self.correct_class.items(), lambda d: (d[1][0], d[1][1]), reverse=state)})
                return classify
            else:
                raise ValueError("`lambda` Caused with `by` Configuration Supports: numbers, accuracy, num-total.")
        else:
            regress.update(loss_)
            return regress

    def _set_container(self, backend: str, n_jobs: int) -> None:
        '''
        Acquire Validation Container with `parallel_backend` at `n_jobs`.
        :param backend: str, "threading", "multiprocessing, 'locky'. default: "threading".
        :param n_jobs: int, set jobs for backend.
        '''
        with parallel_backend(backend, n_jobs=n_jobs):
            val_iter, self.val_container = iter(self.val_loader), []
            while 1:
                try:
                    self.val_container.append(next(val_iter))
                except StopIteration:
                    break

def train_test_val_split(features: TabularData, target: TabularData, ratio_set: Dict[str, int], random_seed: Optional[int]) -> Tuple[Dict[str, TabularData]]:
    '''
    Split TabularData into train, test, val with `ratio_set` and `random_seed`.
    :param features: TabularData, composed of n-row samples and m-column features.
    :param target: TabularData, consists of correct labels or values with size at n.
    :param ratio_set: Dict[str, int], stores the proportion of train-set, test-set and val-set.
    :param random_seed: int | None, random.seed(random_seed) used for fixed random datasets.

    :return train_dict, test_dict, val_dict: Tuple[Dict[str, TabularData]], dataset stores in dictionary.
    '''
    num_examples: int = len(features) # get length of samples
    num_train: int = int(ratio_set['train'] * 0.1 * num_examples)
    num_test: int = int(ratio_set['test'] * 0.1 * num_examples)
    num_val: int = int(ratio_set['val'] * 0.1 * num_examples)
    indices: list = list(range(num_examples))
    train_dict, test_dict, val_dict = dict(), dict(), dict()
    if random_seed != None:
        random.seed(random_seed)
    random.shuffle(indices)
    indices_train = numpy.array(indices[:num_train])
    indices_test = numpy.array(indices[num_train: num_train + num_test])
    indices_val = numpy.array(indices[num_train + num_test: num_train + num_test + num_val])
    train_dict['features'], train_dict['target'] = features[indices_train], target[indices_train]
    test_dict['features'], test_dict['target'] = features[indices_test], target[indices_test]
    val_dict['features'], val_dict['target'] = features[indices_val], target[indices_val]
    return train_dict, test_dict, val_dict