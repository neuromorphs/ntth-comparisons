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
The default pretrained network used is composed by DNN with 4 neuron in the input layer, 4 hidden layer with 128 ReLU neuron each and classification layer composed by 3 neuron with Softmax activation function, no bias is used.
```
conda activate ntth
python main_nest.py
```




