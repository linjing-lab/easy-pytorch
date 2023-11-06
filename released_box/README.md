# perming

perming: Perceptron Models Are Training on Windows Platform with Default GPU Acceleration.

- p: use polars or pandas to read dataset.
- per: perceptron algorithm used as based model.
- m: models include Box, Regressier, Binarier, Mutipler and Ranker.
- ing: training on windows platform with strong gpu acceleration.

## init backend
 
refer to https://pytorch.org/get-started/locally/ and choose PyTorch to support `cuda` compatible with your Windows.

tests with: PyTorch 1.7.1+cu101

## advices

- If users don't want to encounter *CUDA out of memory* return from *joblib.parallel*, the best solution is to download versions before v1.6.1.
- If users have no plan to retrain a full network in tuning model, the best solution is to download versions after v1.8.0 which support *set_freeze*.
- If users are not conducting experiments on Jupyter, download versions after v1.7.* will accelerate *train_val* process and reduce redundancy.

## parameters

init:
- input_: *int*, feature dimensions of tabular datasets after extract, transform, load from any data sources.
- num_classes: *int*, define numbers of classes or outputs after users defined the type of task with layer output.
- hidden_layer_sizes: *Tuple[int]=(100,)*, define numbers and sizes of hidden layers to enhance model representation.
- device: *str='cuda'*, configure training and validation device with torch.device options. 'cuda' or 'cpu'.
- activation: *str='relu'*, configure activation function combined with subsequent learning task. see _activate in open models.
- inplace_on: *bool=False*, configure whether to enable inplace=True on activation. False or True. (manually set in Box)
- criterion: *str='CrossEntropyLoss'*, configure loss criterion with compatible learning task output. see _criterion in open models.
- solver: *str='adam'*, configure inner optimizer serve as learning solver for learning task. see _solver in _utils/BaseModel.
- batch_size: *int=32*, define batch size on loaded dataset of one epoch training process. any int value > 0. (prefer 2^n)
- learning_rate_init: *float=1e-2*, define initial learning rate of solver input param controled by inner assertion. (1e-6, 1.0).
- lr_scheduler: *Optional[str]=None*, configure scheduler about learning rate decay for compatible use. see _scheduler in _utils/BaseModel.

data_loader:
- features: *TabularData*, manually input by users.
- target: *TabularData*, manually input by users.
- ratio_set: *Dict[str, int]={'train': 8, 'test': 1, 'val': 1}*, define by users.
- worker_set: *Dict[str, int]={'train': 8, 'test': 2, 'val': 1}*, manually set by users need.
- random_seed: *Optional[int]=None*, manually set any int value by users to fixed sequence.

set_freeze:
- require_grad: *Dict[int, bool]*, manually set freezed layers by given serial numbers according to `self.model`. (if users set require_grad with `{0: False}`, it means freeze the first layer of `self.model`.)

train_val:
- num_epochs: *int=2*, define numbers of epochs in main training cycle. any int value > 0.
- interval: *int=100*, define console print length of whole epochs by interval. any int value > 0.
- tolerance: *float=1e-3*, define tolerance used to set inner break sensitivity. (1e-9, 1.0).
- patience: *int=10*, define value coordinate with tolerance to expand detect length. [10, 100].
- backend: *str='threading'*, configure accelerate backend used in inner process. 'threading', 'multiprocessing', 'loky'.
- n_jobs: *int=-1*, define numbers of jobs with manually set by users need. -1 or any int value > 0. (if n_jobs=1, parallel processing will be turn off to save cuda memory.)
- early_stop: *bool=False*, define whether to enable early_stop process. False or True.

test:
- sort_by: *str='accuracy'*, configure sorted ways of correct_class. 'numbers', 'accuracy', 'num-total'.
- sort_state: *bool=True*, configure sorted state of correct_class. False or True.

save or load:
- con: *bool=True*, configure whether to print model.state_dict(). False or True.
- dir: *dir='./model'*, configure model path that *save to* or *load from*. correct path defined by users.

## general model

|GENERAL_BOX(Box)|Parameters|Meaning|
|--|--|--|
|`__init__`|input_: int<br />num_classes: int<br />hidden_layer_sizes: Tuple[int]=(100,)<br />device: str='cuda'<br />*<br />activation: str='relu'<br />inplace_on: bool=False<br />criterion: str='CrossEntropyLoss'<br />solver: str='adam'<br />batch_size: int=32<br />learning_rate_init: float=1e-2<br />lr_scheduler: Optional[str]=None|Initialize Classifier or Regressier Based on Basic Information of the Dataset Obtained through Data Preprocessing and Feature Engineering.|
|print_config|/|Return Initialized Parameters of Multi-layer Perceptron and Graph.|
|data_loader|features: TabularData<br />labels: TabularData<br />ratio_set: Dict[str, int]={'train': 8, 'test': 1, 'val': 1}<br />worker_set: Dict[str, int]={'train': 8, 'test': 2, 'val': 1}<br />random_seed: Optional[int]=None|Using `ratio_set` and `worker_set` to Load the Numpy Dataset into `torch.utils.data.DataLoader`.|
|train_val|num_epochs: int=2<br />interval: int=100<br />tolerance: float=1e-3<br />patience: int=10<br />backend: str='threading'<br />n_jobs: int=-1<br />early_stop: bool=False|Using `num_epochs`, `tolerance`, `patience` to Control Training Process and `interval` to Adjust Print Interval with Accelerated Validation Combined with `backend` and `n_jobs`.|
|test|sort_by: str='accuracy'<br />sort_state: bool=True|Sort Returned Test Result about Correct Classes with `sort_by` and `sort_state` Which Only Appears in Classification.|
|save|con: bool=True<br />dir: str='./model'|Save Trained Model Parameters with Model `state_dict` Control by `con`.|
|load|con: bool=True<br />dir: str='./model'|Load Trained Model Parameters with Model `state_dict` Control by `con`.|

## common models (cuda first)

- Regression

|Regressier|Parameters|Meaning|
|--|--|--|
|`__init__`|input_: int<br />hidden_layer_sizes: Tuple[int]=(100,)<br />*<br />activation: str='relu'<br />criterion: str='MSELoss'<br />solver: str='adam'<br />batch_size: int=32<br />learning_rate_init: float=1e-2<br />lr_scheduler: Optional[str]=None|Initialize Regressier Based on Basic Information of the Regression Dataset Obtained through Data Preprocessing and Feature Engineering with `num_classes=1`.|
|print_config|/|Return Initialized Parameters of Multi-layer Perceptron and Graph.|
|data_loader|features: TabularData<br />labels: TabularData<br />ratio_set: Dict[str, int]={'train': 8, 'test': 1, 'val': 1}<br />worker_set: Dict[str, int]={'train': 8, 'test': 2, 'val': 1}<br />random_seed: Optional[int]=None|Using `ratio_set` and `worker_set` to Load the Regression Dataset with Numpy format into `torch.utils.data.DataLoader`.|
|set_freeze|require_grad: Dict[int, bool]|freeze some layers by given `requires_grad=False` if trained model will be loaded to execute experiments.  |
|train_val|num_epochs: int=2<br />interval: int=100<br />tolerance: float=1e-3<br />patience: int=10<br />backend: str='threading'<br />n_jobs: int=-1<br />early_stop: bool=False|Using `num_epochs`, `tolerance`, `patience` to Control Training Process and `interval` to Adjust Print Interval with Accelerated Validation Combined with `backend` and `n_jobs`.|
|test|/|Test Module Only Show with Loss at 3 Stages: Train, Test, Val|
|save|con: bool=True<br />dir: str='./model'|Save Trained Model Parameters with Model `state_dict` Control by `con`.|
|load|con: bool=True<br />dir: str='./model'|Load Trained Model Parameters with Model `state_dict` Control by `con`.|

- Binary-classification

|Binarier|Parameters|Meaning|
|--|--|--|
|`__init__`|input_: int<br />hidden_layer_sizes: Tuple[int]=(100,)<br />*<br />activation: str='relu'<br />criterion: str='BCELoss'<br />solver: str='adam'<br />batch_size: int=32<br />learning_rate_init: float=1e-2<br />lr_scheduler: Optional[str]=None|Initialize Classifier Based on Basic Information of the Classification Dataset Obtained through Data Preprocessing and Feature Engineering with `num_classes=2`.|
|print_config|/|Return Initialized Parameters of Multi-layer Perceptron and Graph.|
|data_loader|features: TabularData<br />labels: TabularData<br />ratio_set: Dict[str, int]={'train': 8, 'test': 1, 'val': 1}<br />worker_set: Dict[str, int]={'train': 8, 'test': 2, 'val': 1}<br />random_seed: Optional[int]=None|Using `ratio_set` and `worker_set` to Load the Binary-classification Dataset with Numpy format into `torch.utils.data.DataLoader`.|
|set_freeze|require_grad: Dict[int, bool]|freeze some layers by given `requires_grad=False` if trained model will be loaded to execute experiments.  |
|train_val|num_epochs: int=2<br />interval: int=100<br />tolerance: float=1e-3<br />patience: int=10<br />backend: str='threading'<br />n_jobs: int=-1<br />early_stop: bool=False|Using `num_epochs`, `tolerance`, `patience` to Control Training Process and `interval` to Adjust Print Interval with Accelerated Validation Combined with `backend` and `n_jobs`.|
|test|sort_by: str='accuracy'<br />sort_state: bool=True|Test Module con with Correct Class and Loss at 3 Stages: Train, Test, Val|
|save|con: bool=True<br />dir: str='./model'|Save Trained Model Parameters with Model `state_dict` Control by `con`.|
|load|con: bool=True<br />dir: str='./model'|Load Trained Model Parameters with Model `state_dict` Control by `con`.|

- Multi-classification

|Mutipler|Parameters|Meaning|
|--|--|--|
|`__init__`|input_: int<br />num_classes: int<br />hidden_layer_sizes: Tuple[int]=(100,)<br />*<br />activation: str='relu'<br />criterion: str='CrossEntropyLoss'<br />solver: str='adam'<br />batch_size: int=32<br />learning_rate_init: float=1e-2<br />lr_scheduler: Optional[str]=None|Initialize Classifier Based on Basic Information of the Classification Dataset Obtained through Data Preprocessing and Feature Engineering with `num_classes>2`.|
|print_config|/|Return Initialized Parameters of Multi-layer Perceptron and Graph.|
|data_loader|features: TabularData<br />labels: TabularData<br />ratio_set: Dict[str, int]={'train': 8, 'test': 1, 'val': 1}<br />worker_set: Dict[str, int]={'train': 8, 'test': 2, 'val': 1}<br />random_seed: Optional[int]=None|Using `ratio_set` and `worker_set` to Load the Multi-classification Dataset with Numpy format into `torch.utils.data.DataLoader`.|
|set_freeze|require_grad: Dict[int, bool]|freeze some layers by given `requires_grad=False` if trained model will be loaded to execute experiments.  |
|train_val|num_epochs: int=2<br />interval: int=100<br />tolerance: float=1e-3<br />patience: int=10<br />backend: str='threading'<br />n_jobs: int=-1<br />early_stop: bool=False|Using `num_epochs`, `tolerance`, `patience` to Control Training Process and `interval` to Adjust Print Interval with Accelerated Validation Combined with `backend` and `n_jobs`.|
|test|sort_by: str='accuracy'<br />sort_state: bool=True|Sort Returned Test Result about Correct Classes with `sort_by` and `sort_state` Which Only Appears in Classification.|
|save|con: bool=True<br />dir: str='./model'|Save Trained Model Parameters with Model `state_dict` Control by `con`.|
|load|con: bool=True<br />dir: str='./model'|Load Trained Model Parameters with Model `state_dict` Control by `con`.|

- Multi-outputs

|Ranker|Parameters|Meaning|
|--|--|--|
|`__init__`|input_: int<br />num_outputs: int<br />hidden_layer_sizes: Tuple[int]=(100,)<br />*<br />activation: str='relu'<br />criterion: str='MultiLabelSoftMarginLoss'<br />solver: str='adam'<br />batch_size: int=32<br />learning_rate_init: float=1e-2<br />lr_scheduler: Optional[str]=None|Initialize Ranker Based on Basic Information of the Classification Dataset Obtained through Data Preprocessing and Feature Engineering with (n_samples, n_outputs).|
|print_config|/|Return Initialized Parameters of Multi-layer Perceptron and Graph.|
|data_loader|features: TabularData<br />labels: TabularData<br />ratio_set: Dict[str, int]={'train': 8, 'test': 1, 'val': 1}<br />worker_set: Dict[str, int]={'train': 8, 'test': 2, 'val': 1}<br />random_seed: Optional[int]=None|Using `ratio_set` and `worker_set` to Load the Multi-outputs Dataset with Numpy format into `torch.utils.data.DataLoader`.|
|set_freeze|require_grad: Dict[int, bool]|freeze some layers by given `requires_grad=False` if trained model will be loaded to execute experiments.  |
|train_val|num_epochs: int=2<br />interval: int=100<br />tolerance: float=1e-3<br />patience: int=10<br />backend: str='threading'<br />n_jobs: int=-1<br />early_stop: bool=False|Using `num_epochs`, `tolerance`, `patience` to Control Training Process and `interval` to Adjust Print Interval with Accelerated Validation Combined with `backend` and `n_jobs`.|
|test|/|Test Module Only Show with Loss at 3 Stages: Train, Test, Val|
|save|con: bool=True<br />dir: str='./model'|Save Trained Model Parameters with Model `state_dict` Control by `con`.|
|load|con: bool=True<br />dir: str='./model'|Load Trained Model Parameters with Model `state_dict` Control by `con`.|

prefer replace target shape *(n,1)* with shape *(n,)* using `numpy.squeeze(target)`, users can search and combine more predefined options in submodules and its `__doc__` of each open classes.

## pip install

download latest version:
```text
git clone https://github.com/linjing-lab/easy-pytorch.git
cd easy-pytorch/released_box
pip install -e . --verbose
```
download stable version:
```text
pip install perming --upgrade
```
download versions without supported *early_stop*:
```text
pip install perming==1.3.1
```
download versions with supported *early_stop*:
```text
pip install perming>=1.4.1
```
download versions with supported *early_stop* in epoch:
```python
pip install perming>=1.4.2
```
download version without enhancing *Parallel* and *delayed*:
```text
pip install perming==1.6.1
```
download version with enhancing *Parallel* and *delayed*:
```text
pip install perming>=1.7.0
```
download version with supported *set_freeze*:
```text
pip install perming>=1.8.0
```
download version without crash of jupyter kernel:
```text
pip install perming>=1.8.1
```
