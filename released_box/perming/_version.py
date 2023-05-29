# Copyright (c) 2023 linjing-lab

def parse_torch_version(version_info: str):
    '''
    :param version_info: str, version information of PyTorch with `__version__`.

    :return tuple value about `version` & `cu_info`.
    '''
    if version_info.find('cu') != -1:
        version_cu = version_info.split('+')
        version = version_cu[0].split('.')
        return version, version_cu[1]
    else:
        raise Exception('Please refer to https://pytorch.org/get-started/locally/ for PyTorch with cuda version compatible with your windows computer.')