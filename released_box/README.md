# perming

perming: Perceptron Models Are Training on Windows Platform with Default GPU Acceleration.

- p: use polars or pandas to read dataset.
- per: perceptron algorithm used as based model.
- m: models concluding regressier and classifier (binary & multiple).
- ing: training on windows platform with strong gpu acceleration.

## init backend
 
refer to https://pytorch.org/get-started/locally/ and choose the PyTorch that support `cuda` compatible with your Windows. The current software version only supports Windows system.

test with: PyTorch 1.7.1+cu101

## general model

|GENERAL_BOX(Box)|Parameters|Meaning|
|--|--|--|
|`__init__`|input_: int<br />num_classes: int<br />hidden_layer_sizes: Tuple[int]=(100,)<br />device: str="cuda"<br />activation: str="relu"<br />inplace_on: bool=True<br />criterion: str="CrossEntropyLoss"<br />solver: str="adam"<br />batch_size: int=32<br />learning_rate_init: float=1e-3<br />lr_scheduler: Optional[str]=None|Initialize Classifier or Regressier Based on Basic Information of the Dataset Obtained through Data Preprocessing and Feature Engineering.|
|print_config|/|Return Initialized Parameters of Multi-layer Perceptron and Graph.|
|data_loader|features: TabularData<br />labels: TabularData<br />ratio_set: Dict[str, int]={'train': 8, 'test': 1, 'val': 1}<br />worker_set: Dict[str, int]={'train': 8, 'test': 2, 'val': 1}<br />random_seed: Optional[int]=None|Using `ratio_set` and `worker_set` to Load the Numpy Dataset into `torch.utils.data.DataLoader`.|
|train_val|num_epochs: int=5<br />interval: int=100<br />backend: str="threading"<br />n_jobs: int=-1|Using `num_epochs` to Control Training Process and `interval` to Adjust Print Interval with Accelerated Validation Combined with `backend` and `n_jobs`.|
|test|sort_by: str="accuracy"<br />sort_kernel: str="bubble"<br />sort_state: bool=True|Sort Returned Test Result about Correct Classes with `sort_by`, `sort_kernel`, `sort_state` Which Only Appears in Classification.|
|save|show: bool=True<br />dir: str='./model'|Save Trained Model Parameters with Model `state_dict` Control by `show`.|
|load|show: bool=True<br />dir: str='./model'|Load Trained Model Parameters with Model `state_dict` Control by `show`.|

## common models (cuda first)

- Regression

|Regressier|Parameters|Meaning|
|--|--|--|
|`__init__`|input_: int<br />hidden_layer_sizes: Tuple[int]=(100,)<br />activation: str="relu"<br />criterion: str="MSELoss"<br />solver: str="adam"<br />batch_size: int=32<br />learning_rate_init: float=1e-3<br />lr_scheduler: Optional[str]=None|Initialize Regressier Based on Basic Information of the Regression Dataset Obtained through Data Preprocessing and Feature Engineering with `num_classes=1`.|
|print_config|/|Return Initialized Parameters of Multi-layer Perceptron and Graph.|
|train_val|num_epochs: int=5<br />interval: int=100<br />backend: str="threading"<br />n_jobs: int=-1|Using `ratio_set` and `worker_set` to Load the Regression Dataset with Numpy format into `torch.utils.data.DataLoader`.|
|test|/|Test Module Only Show with Loss at 3 Stages: Train, Test, Val|
|save|show: bool=True<br />dir: str='./model'|Save Trained Model Parameters with Model `state_dict` Control by `show`.|
|load|show: bool=True<br />dir: str='./model'|Load Trained Model Parameters with Model `state_dict` Control by `show`.|

- Binary-classification

|Binarier|Parameters|Meaning|
|--|--|--|
|`__init__`|input_: int<br />hidden_layer_sizes: Tuple[int]=(100,)<br />activation: str="relu"<br />criterion: str="BCELoss"<br />solver: str="adam"<br />batch_size: int=32<br />learning_rate_init: float=1e-3<br />lr_scheduler: Optional[str]=None|Initialize Classifier Based on Basic Information of the Classification Dataset Obtained through Data Preprocessing and Feature Engineering with `num_classes=2`.|
|print_config|/|Return Initialized Parameters of Multi-layer Perceptron and Graph.|
|train_val|num_epochs: int=2<br />interval: int=100<br />backend: str="threading"<br />n_jobs: int=-1|Using `ratio_set` and `worker_set` to Load the Regression Dataset with Numpy format into `torch.utils.data.DataLoader`.|
|test|sort_by: str="accuracy"<br />sort_kernel: str="bubble"<br />sort_state: bool=True|Test Module Show with Correct Class and Loss at 3 Stages: Train, Test, Val|
|save|show: bool=True<br />dir: str='./model'|Save Trained Model Parameters with Model `state_dict` Control by `show`.|
|load|show: bool=True<br />dir: str='./model'|Load Trained Model Parameters with Model `state_dict` Control by `show`.|

- Multi-classification

|Multipler|Parameters|Meaning|
|--|--|--|
|`__init__`|input_: int<br />num_classes: int<br />hidden_layer_sizes: Tuple[int]=(100,)<br />activation: str="relu"<br />criterion: str="CrossEntropyLoss"<br />solver: str="adam"<br />batch_size: int=32<br />learning_rate_init: float=1e-3<br />lr_scheduler: Optional[str]=None|Initialize Classifier Based on Basic Information of the Classification Dataset Obtained through Data Preprocessing and Feature Engineering with `num_classes>2`.|
|print_config|/|Return Initialized Parameters of Multi-layer Perceptron and Graph.|
|train_val|num_epochs: int=2<br />interval: int=100<br />backend: str="threading"<br />n_jobs: int=-1|Using `ratio_set` and `worker_set` to Load the Regression Dataset with Numpy format into `torch.utils.data.DataLoader`.|
|test|sort_by: str="accuracy"<br />sort_kernel: str="bubble"<br />sort_state: bool=True|Sort Returned Test Result about Correct Classes with `sort_by`, `sort_kernel`, `sort_state` Which Only Appears in Classification.|
|save|show: bool=True<br />dir: str='./model'|Save Trained Model Parameters with Model `state_dict` Control by `show`.|
|load|show: bool=True<br />dir: str='./model'|Load Trained Model Parameters with Model `state_dict` Control by `show`.|

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