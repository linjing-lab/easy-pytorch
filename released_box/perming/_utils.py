# Copyright (c) 2023 linjing-lab

import torch, random, numpy, gc
from joblib import parallel_backend
from collections import OrderedDict
from ._typing import TabularData, Tuple, Dict, Optional, Any
from ._version import parse_torch_version

class TabularDataset(torch.utils.data.Dataset):
    '''
    Tabular Data Constructed with `numpy.array` noted as `TabularData`.
    :param features: TabularData, composed of n-row samples and m-column features.
    :param target: TabularData, consists of correct labels or outputs with size at n.
    :param roc: bool, as_tensor to torch.float or torch.long. select torch.float when `roc=True`.
    '''
    def __init__(self, features: TabularData, target: TabularData, roc: bool) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float)
        self.target = torch.as_tensor(target, dtype=torch.float) if roc else torch.as_tensor(target, dtype=torch.long)

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self) -> int:
        return len(self.features)

class MLP(torch.nn.Module):
    '''
    Construct Model Layers with `input_`, `num_classes`, `hidden_layer_sizes`, `activation`.
    :param input_: int, input dataset with features' dimension of tabular data is input_.
    :param num_classes: int, total number of correct label categories or multi-outputs.
    :param hidden_layer_sizes: Tuple[int], configure the length and size of each hidden layer.
    :param activation:, activation configured by Box, Regressier, Binarier, Multipler, and Ranker.
    '''
    def __init__(self, input_: int, num_classes: int, hidden_layer_sizes: Tuple[int], activation) -> None:
        super(MLP, self).__init__()
        self.squeeze: bool = False if num_classes > 1 else True
        if hidden_layer_sizes: # for linear indivisible datasets
            assert hidden_layer_sizes[0] > 0, 'Please ensure any layer of hidden_layer_sizes > 0'
            model_layers, hidden_length = OrderedDict(), len(hidden_layer_sizes)
            model_layers['Linear0'] = torch.nn.Linear(input_, hidden_layer_sizes[0])
            model_layers['Activation0'] = activation
            for index in range(1, hidden_length):
                assert hidden_layer_sizes[index] > 0, 'Please ensure any layer of hidden_layer_sizes > 0'
                ind_str = str(index)
                linear_, activation_ = 'Linear'.join(('', ind_str)), 'Activation'.join(('', ind_str))
                model_layers[linear_] = torch.nn.Linear(hidden_layer_sizes[index - 1], hidden_layer_sizes[index])
                model_layers[activation_] = activation
            model_layers['Linear'.join(('', str(hidden_length)))] = torch.nn.Linear(hidden_layer_sizes[hidden_length - 1], num_classes)
            self.mlp = torch.nn.Sequential(model_layers)
        else:
            self.mlp = torch.nn.Linear(input_, num_classes)

    def forward(self, x):
        output = self.mlp(x)
        return output.squeeze(-1) if self.squeeze else output

class BaseModel:
    '''
    Basic Model for Configuring Common Models and General Box.
    :param input_: int, input dataset with features' dimension of tabular data is input_.
    :param num_classes: int, total number of correct label categories or multi-outputs.
    :param hidden_layer_sizes: Tuple[int], configure the size of each hidden layer.
    :param device: Any, device configured by common models and general box.
    :param activation: Any, activated function configured by common models and general box.
    :param criterion: Any, loss function determined by common models and general box.
    :param solver: str, optimization coordinated with `torch.optim.lr_scheduler`. (modified more params in `_solver`.)
    :param batch_size: int, batch size of tabular dataset in one training process.
    :param learning_rate_init: float, initialize the learning rate of the optimizer.
    :param lr_scheduler: str | None, set the learning rate scheduler integrated with the optimizer. default: None. (modified more params in `_scheduler`.)
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
        assert input_ > 0, 'BaseModel need samples with dataset features named input_ > 0.'
        assert num_classes > 0, 'Supervised learning problems with num_classes ranges from (1, 2, 3, ...).'
        assert batch_size > 0, 'Batch size initialized with int value mostly 2^n(n=1, 2, 3), like 64, 128, 256.'
        assert learning_rate_init > 1e-6 and learning_rate_init < 1.0, 'Please assert learning rate initialized value in (1e-6, 1.0).'
        self.input: int = input_
        self.num_classes: int = num_classes
        self.activation = activation # function activate high-dimensional features
        self.device = device # device configuration
        self.criterion = criterion # criterion with classification & torch.long, regression & torch.float, and multi-outputs & roc
        self.batch_size: int = batch_size
        self.lr: float = learning_rate_init
        self.model = MLP(self.input, self.num_classes, hidden_layer_sizes, self.activation).to(self.device)
        if parse_torch_version(torch.__version__)[0] >= ['2', '0', '0']: # compile model
            self.model = torch.compile(self.model)
        self.solver = self._solver(solver)
        self.lr_scheduler = self._scheduler(lr_scheduler)

    def _solver(self, solver: str):
        '''
        Configure Optimizer with `solver`.
        :param solver: str, 'sgd', 'momentum', 'adam', 'adagrad', 'rmsprop'. default: adam.
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
        :param lr_scheduler: str, 'exponential_lr', 'step_lr', 'cosine_annealing_lr'. default: None.
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
        Encapsulate Dataset to DataLoader from Tabular Dataset.
        :param features: TabularData, composed of n-row samples and m-column features: (n_samples, n_features).
        :param target: TabularData, consists of correct labels or values: (n_samples,) or (n_samples, n_outputs).
        :param ratio_set: Dict[str, int], stores the proportion of train-set, test-set and val-set. default: {'train': 8, 'test': 1, 'val': 1}.
        :param worker_set: Dict[str, int], load the num_workers of DataLoader with multi-threaded. default: {'train': 8, 'test': 2, 'val': 1}.
        :param random_seed: int | None, random.seed(random_seed) used for fixed random datasets. default: None.
        '''
        assert ratio_set['train'] > 0 and ratio_set['test'] > 0 and ratio_set['val'] > 0
        assert ratio_set['train'] > 4 * ratio_set['test'], 'The training set needs to be larger than the test set.'
        assert ratio_set['train'] + ratio_set['test'] + ratio_set['val'] == 10, 'The sum of 3 datasets ratio needs to be 10.'
        assert isinstance(features, TabularData) and features.ndim == 2, 'Please ensure features with dimension at (n_samples, n_features).'
        assert features.shape[1] == self.input, 'Please ensure `input_` is equal to `features.shape[1]`.'
        assert isinstance(target, TabularData), 'Please ensure target format at numpy.ndarray noted as TabularData.'
        target_dtype = str(target.dtype) # __str__
        is_int_type, is_float_type = 'int' in target_dtype, 'float' in target_dtype
        self.is_target_2d = isinstance(target[0], TabularData) # judge target is 2d matrix.
        self.is_task_c1d = not self.is_target_2d and self.num_classes >= 2 # if task is 1d classification
        if self.is_target_2d: # (n_samples, n_outputs)
            assert target.shape[1] == self.num_classes, 'Please ensure target with (n_samples, n_outputs=num_classes).'
            assert target.shape[1] >= 1, 'Please convert (n,1) to (n,) with numpy.squeeze(target) then explore type_of_problems.'
            assert is_int_type or is_float_type, 'Please ensure target.dtype in any int or float type of numpy.dtype.'
            roc: bool = not is_int_type and is_float_type
        else: # (n_samples,)
            if self.num_classes >= 2:
                self.unique = numpy.unique(target) # int indexes -> any class with single value noted
                assert len(self.unique) == self.num_classes, 'Please ensure `num_classes` is equal to `len(numpy.unique(labels))`.'
                self.indices = dict(zip(self.unique, range(self.num_classes))) # original classes -> int indexes
                target = numpy.array([self.indices[value] for value in target], dtype=numpy.int8) # adjust int8 -> any int dtype
            else:
                assert is_float_type, 'Please ensure target.dtype in any float type of numpy.dtype.' # continuous
            roc: bool = self.model.squeeze
        train_, test_, val_ = train_test_val_split(features, target, ratio_set, random_seed)
        self.train_loader = torch.utils.data.DataLoader(TabularDataset(train_['features'], train_['target'], roc), batch_size=self.batch_size, shuffle=True, num_workers=worker_set['train'])
        self.test_loader = torch.utils.data.DataLoader(TabularDataset(test_['features'], test_['target'], roc), batch_size=self.batch_size, shuffle=True, num_workers=worker_set['test'])
        self.val_loader = torch.utils.data.DataLoader(TabularDataset(val_['features'], val_['target'], roc), batch_size=self.batch_size, shuffle=False, num_workers=worker_set['val'])

    def train_val(self, 
                  num_epochs: int=2,
                  interval: int=100,
                  tolerance: float=1e-3,
                  patience: int=10, 
                  backend: str='threading', 
                  n_jobs: int=-1,
                  early_stop: bool=False) -> None:
        '''
        Training and Validation with `train_loader` and `val_container`.
        :param num_epochs: int, training epochs for `self.model`. default: 2.
        :param interval: int, console output interval. default: 100.
        :param tolerance: float, tolerance set to judge difference in val_loss. default: 1e-3
        :param patience: int, patience of no improvement waiting for training to stop. default: 10.
        :param backend: str, 'threading', 'multiprocessing', 'locky'. default: 'threading'.
        :param n_jobs: int, accelerate processing of validation. default: -1.
        :param early_stop: bool, whether to enable early_stop in train_val. default: False.
        '''
        assert num_epochs > 0 and interval > 0, 'With num_epochs > 0 and interval > 0 to train parameters into outputs.'
        assert tolerance > 1e-9 and tolerance < 1.0, 'Set tolerance to early stop training and validation process within patience.'
        assert patience >= 10 and patience <= 100, 'Value coordinate with tolerance should fit about num_epochs and batch_size.' 
        assert n_jobs == -1 or n_jobs > 0, 'Take full jobs with setting n_jobs=-1 or manually set nums of jobs.'
        total_step = len(self.train_loader)
        self._set_container(backend, n_jobs)
        val_length: int = len(self.val_container)
        self.stop_iter: bool = False # init state of train_val
        for epoch in range(num_epochs):
            gc.collect()
            torch.cuda.empty_cache()
            for i, (features, target) in enumerate(self.train_loader):
                features = features.to(self.device)
                target = target.to(self.device)

                # forward pass
                outputs = self.model(features)
                self.train_loss = self.criterion(outputs, target)

                # backward and optimize
                self.solver.zero_grad()
                self.train_loss.backward()
                _ = self.solver.step() if self.lr_scheduler == None else self.lr_scheduler.step()

                # validation with val_container
                self.val_loss = 0 # int value at cpu
                with parallel_backend(backend, n_jobs=n_jobs):
                    for val_set in self.val_container:
                        outputs_val = self.model(val_set[0].to(self.device)) # return value from cuda
                        self.val_loss += self.criterion(outputs_val, val_set[1].to(self.device))
                self.val_loss /= val_length
                val_counts = i + 1 + total_step * epoch # times of val_loss renewed

                # early stop
                if early_stop:
                    if val_counts == 1: # record first time of val_loss
                        val_loss_pre, val_pos_ini = self.val_loss, 1
                    else:
                        if val_loss_pre < self.val_loss:
                            val_loss_pre, val_pos_ini = self.val_loss, val_counts
                        else:
                            if (val_counts - val_pos_ini + 1) == patience:
                                if val_loss_pre - self.val_loss < tolerance:
                                    self.stop_iter: bool = True
                                    break
                                else:
                                    val_loss_pre, val_pos_ini = self.val_loss, val_counts
                # console print
                if (i + 1) % interval == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, self.train_loss.item(), self.val_loss.item()))

            if self.stop_iter:
                print('Process stop at epoch [{}/{}] with patience {} within tolerance {}'.format(epoch+1, num_epochs, patience, tolerance))
                break

    def test(self, 
             sort_by: str='accuracy',
             sort_state: bool=True):
        '''
        Configured keywords only work when `not self.is_target_2d and num_classes >= 2`.
        Produce `self.aver_acc != 0` and 'correct_class != None' in the above condition.
        :param sort_by: str, 'accuracy', 'numbers', 'num-total'. default: 'accuracy'.
        :param sort_state: bool, whether to use descending order when sorting. default: True.
        '''
        with torch.no_grad():
            self.test_loss, test_loader_step, correct = 0, len(self.test_loader), 0
            test_total = test_loader_step * self.batch_size
            self.correct_class = dict.fromkeys(self.unique, [0, 0]) if self.is_task_c1d else None
            for features, target in self.test_loader:
                features = features.to(self.device)
                target = target.to(self.device)            
                outputs = self.model(features)
                if self.is_task_c1d:
                    _, predicted = torch.max(outputs.data, 1)
                    for index, value in enumerate(predicted):
                        self.correct_class[self.unique[value]][1] += 1 # record total numbers of each class
                        self.correct_class[self.unique[value]][0] += (value == target[index]).item() # record total true numbers of each class
                    correct += (predicted == target).sum().item()
                self.test_loss += self.criterion(outputs, target)
            self.test_loss /= test_loader_step
            self.aver_acc = 100 * correct / test_total if self.is_task_c1d else None
            print('loss of {0} on the {1} test dataset: {2}.'.format(self.__class__.__name__, test_total, self.test_loss.item()))
        return OrderedDict(self._pack_info(sort_by, sort_state))

    def save(self, show: bool=True, dir: str='./model') -> None:
        '''
        Save Model Checkpoint with Box, Regressier, Binarier, Multipler, and Ranker.
        :param show: bool, whether to show `model.state_dict()`. default: True.
        :param dir: str, model save to dir. default: './model'.
        '''
        if show:
            print(self.model.state_dict())
        torch.save(self.model.state_dict(), dir)

    def load(self, show: bool=True, dir: str='./model') -> None:
        '''
        Load Model Checkpoint with Box, Regressier, Binarier, Multipler, and Ranker.
        :param show: bool, whether to show `model.state_dict()`. default: True.
        :param dir: str, model load from dir. default: './model'.
        '''
        params = torch.load(dir)
        self.model.load_state_dict(params)
        if show:
            print(self.model.state_dict())

    def _pack_info(self, by: str, state: bool) -> Dict[str, Any]:
        '''
        Pack Test Returned Data including `correct_class` and returned result of `sorted`.
        :param by: str, choose which way to sort the order of 'correct_class'.
        :param state: bool, choose the state when `correct_class` is sorting.
        '''
        loss_, classify, regress, outputs = {
            'loss': {'train': self.train_loss.item(), 
                     'val': self.val_loss.item(),
                     'test': self.test_loss.item()}
        }, {'problem': 'classification',
            'accuracy': f'{self.aver_acc}%',
            'num_classes': self.num_classes,
            'column': ('label name', ('true numbers', 'total numbers')),
            'labels': self.correct_class}, {'problem': 'regression'}, {'problem': 'multi-outputs'}
        if self.is_task_c1d:
            classify.update(loss_)
            if by == 'numbers':
                classify.update({'sorted': sorted(self.correct_class.items(), key=lambda d: d[1][0], reverse=state)})
                return classify
            elif by == 'accuracy':
                classify.update({'sorted': sorted(self.correct_class.items(), key=lambda d: d[1][0]/d[1][1], reverse=state)})
                return classify
            elif by == 'num-total':
                classify.update({'sorted': sorted(self.correct_class.items(), key=lambda d: (d[1][0], d[1][1]), reverse=state)})
                return classify
            else:
                raise ValueError("`lambda` Caused with `by` Configuration Supports: numbers, accuracy, num-total.")
        else:
            if self.is_target_2d:
                outputs.update(loss_)
                return outputs # Multi-outputs
            else:
                regress.update(loss_)
                return regress

    def _set_container(self, backend: str, n_jobs: int) -> None:
        '''
        Validation Container with `parallel_backend` at `n_jobs`.
        :param backend: str, "threading", "multiprocessing, 'locky'.
        :param n_jobs: int, set jobs with backend to accelerate process.
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