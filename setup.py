
import io
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Package meta-data.
NAME = 'st-graphpca'
DESCRIPTION = 'A fast and interpretable dimension reduction algorithm for spatial transcriptomics data.'
EMAIL = '599568651@qq.com'
URL="https://github.com/YANG-ERA/GraphPCA/tree/master"
AUTHOR ='Jiyuan Yang'
VERSION = '0.0.4'

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
	license='MIT',
    description=DESCRIPTION,
	url=URL,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.6",
        "pandas==2.0.1",
        "matplotlib==3.7.3",
        "scipy==1.9.1",
        "scikit-learn==0.24.0",
        "networkx==2.8.4",
        "scanpy==1.9.6",
        "squidpy==1.2.3"
    ]
)
