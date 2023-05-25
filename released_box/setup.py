import os
import sys

from setuptools import setup

if sys.version_info < (3, 7, 0):
    raise OSError(f'perming requires Python >=3.7, but yours is {sys.version}')

try:
    pkg_name = 'perming'
    libinfo_py = os.path.join(pkg_name, '__init__.py')
    libinfo_content = open(libinfo_py, 'r', encoding='utf-8').readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
    exec(version_line) # gives __version
except FileNotFoundError:
    __version__ = '0.0.0'

try:
    with open('README.md', 'r', encoding='utf-8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
    name=pkg_name,
    packages=['perming',],
    version=__version__,
    description='The supervised learning framework based on perceptron for tabular data.',
    author='林景',
    author_email='linjing010729@163.com',
    license='MPL 2.0',
    url='https://github.com/linjing-lab/easy-pytorch/tree/main/released_box',
    download_url='https://github.com/linjing-lab/easy-pytorch/tags',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    zip_safe=False,
    setup_requires=['setuptools>=18.0', 'wheel'],
    project_urls={
        'Source': 'https://github.com/linjing-lab/easy-pytorch/tree/main/released_box/perming',
        'Tracker': 'https://github.com/linjing-lab/easy-pytorch/issues',
    },
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires = [
        "polars[pandas]>=0.17.0",
        "numpy>=1.21.0",
        'sortingx>=1.2.2',
        'joblib>=1.1.0'
    ]
    # extras_require = []
)