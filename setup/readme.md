massibo-ana uses python 3.10.12 and the following third-party packages

- numpy 1.26.1
- pandas 2.1.1
- scipy 1.11.3
- matplotlib 3.8.0
- json5 0.9.14

It is recommended to set up a virtual environment particularly for massibo-ana, in which case, you can use `pip` to install all of the necessary packages in your virtual environment by running

```bash
pip install -r requirements.txt
```

within the virtual environment. The `requirements.txt` file, which is the one provided in this directory, will also install jupyter 1.0.0.

Additionally, running the `setup.sh` script which is allocated in this directory will set an environment variable called `MASSIBOANAPATH` to the root folder of the repository. For the time being, the only responsibility of such script is to set such environment variable. If it is not working for your system, you may write yours.
