# Description
Tools for cleaning, analyzing and graphing output data from Massibo, which is a setup for cryogenic characterization of SiPMs' dark noise rate and gain

massibo_ana uses python 3.10.12 and the following third-party packages

- numpy 1.26.1
- pandas 2.1.1
- scipy 1.11.3
- matplotlib 3.8.0
- json5 0.9.14

# Set up
It is recommended to set up a virtual environment particularly for massibo_ana. Assuming you are already running within the desired virutal environment, navigate to the massibo_ana root directory and build the project by running

```bash
python -m build
```

To this end, you might need to install the `build` package within your virtual environment. You can do so by running `pip install build`. Once the project has been built, you can install it by running

```bash
pip install -r requirements.txt
````

The command above will install the necessary dependencies for massibo_ana, as well as the massibo_ana package itself. If you intend to develop within massibo_ana, it may be convenient to install the massibo_ana package in editing-mode, so that changes in the code are effective in the environment without needing to re-install massibo_ana. To do so, run 

```bash
pip install -e .
```

in the massibo_ana root directory.