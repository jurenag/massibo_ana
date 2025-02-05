# massibo_ana
Tools for cleaning, analyzing and graphing output data from Massibo, which is a setup for cryogenic characterization of SiPMs' dark noise rate and gain

massibo_ana uses python 3.10.12 and the following third-party packages

- numpy 1.26.1
- pandas 2.1.1
- scipy 1.11.3
- matplotlib 3.8.0
- json5 0.9.14

It is recommended to set up a virtual environment particularly for massibo_ana, in which case, you can use `pip` to install all of the necessary packages in your virtual environment by running

```bash
pip install -r requirements.txt
```

within the virtual environment. The `requirements.txt` file, which is the one provided in this directory, will also install jupyter 1.0.0.