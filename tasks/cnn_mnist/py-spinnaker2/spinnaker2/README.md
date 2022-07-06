# Required software
- python
- `gcc-arm-none-eabi`
- [digilent runtime](https://files.digilent.com/Software/Adept2+Runtime/2.20.2/digilent.adept.runtime_2.20.2-x86_64.tar.gz)
- PCL library (`libpcl`)
- CMake

# Required python packages
Install the required python packages to your virtual environment: `pip install -r requirements.txt`

# Download and install the Spinnaker 2 libraries
Make sure you have access to the `s2-sim2lab-app` repository, then run: `python install.py`

# Run the test notebook
The test is in notebook `cnn_mnist.ipynb`.

In order to make your python environment (aka `kernel`) visible to jupyter, you can do:
1. Anaconda:
```
conda install nb_conda
```
2. Virtualenv called `myenv` (not tested):
```
pip install ipykernel
python -m ipykernel install --name=myenv
```
Then select the kernel of your choice in the jupyter notebook.
