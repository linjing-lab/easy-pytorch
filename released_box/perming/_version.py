# Copyright (c) 2023 linjing-lab

def parse_torch_version(version_info: str):
    '''
    :param version_info: str, version information of PyTorch with `__version__`.

    :return tuple value about `version` & `cu_info`.
    '''
    if version_info.find('cu') != -1:
        version_cu = version_info.split('+')
        version = tuple(version_cu[0].split('.'))
        cuda_info = version_cu[1]
        return version, cuda_info
    else:
        raise Exception('Please refer to https://pytorch.org/get-started/locally/ for PyTorch with cuda version compatible with your windows computer.')