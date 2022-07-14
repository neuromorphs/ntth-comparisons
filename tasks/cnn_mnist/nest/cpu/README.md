# Required software
- Python
- Conda
- [NEST Simulator](https://www.nest-simulator.org/)
- [PyNN](https://neuralensemble.org/PyNN/)

# Getting started

## Installation
We highly recommend to install this software inside a virtual environment
### Conda virtual environment
1. Conda installation:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
chmod 777 Miniconda3-py38_4.10.3-Linux-x86_64.sh
./Miniconda3-py38_4.10.3-Linux-x86_64.sh
```
2. Conda virtual enviroments creation:
```
conda create -n ntth
conda activate ntth
```
3. Installation Nest-simulator and PyNN
```
conda install -c conda-forge nest-simulator==2.20.1
conda install pip
pip install --upgrade pip
pip install pynn==0.9.6
```
4. Install required packages
```
pip install tensorflow==2.9.1
pip install tensorflow-datasets=4.6.0
pip install matplotlib==3.5.2
pip install seaborn==0.11.2
```

## Run the simulation
The default pretrained network used is composed by CNN with 2 convolutional layer with 6 filter each, with 3x3 kernel size with custom activation function [sReLU](https://doi.org/10.48550/arXiv.1706.03609), followed by average pooling layer with 2x2 pool size layer, for the classification layer is used one hidden layer with 768 units and finally the classification layer with 10 output neuron and softmax activation function, in all network no bias is used.
```
conda activate ntth
python main_nest.py
```




