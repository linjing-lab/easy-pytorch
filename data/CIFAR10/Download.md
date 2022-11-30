# DOWNLOAD
```python
train_dataset = torchvision.datasets.CIFAR10('data/CIFAR/', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10('data/CIFAR/', train=False, download=True, transform=transform_test)
```