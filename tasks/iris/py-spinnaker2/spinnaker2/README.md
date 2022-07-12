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
conda create --name spinnaker2 python=3.8
conda activate spinnaker2
```
3. Install required packages
```
conda install pip
pip install --upgrade pip
pip install scipy
pip install tensorflow==2.9.1
pip install tensorflow-datasets=4.6.0
pip install h5py==3.6 # required for tf_datasets
pip install matplotlib==3.5.2
pip install seaborn==0.11.2
```
5. Install Python module `spinnaker2`
```
git clone git@gitlab.com:spinnaker2/py-spinnaker2.git
cd py-spinnaker2
pip install -e .

```

## Run the simulation
The default pretrained network used is composed by DNN with 4 neuron in the input layer, 4 hidden layer with 128 ReLU neuron each and classification layer composed by 3 neuron with Softmax activation function, no bias is used.
```
conda activate ntth
python main_spinnaker2.py
```




