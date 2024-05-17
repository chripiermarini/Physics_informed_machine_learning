# SQPPIML
### Step[0] (Skip if Python3 (>=3.12) is installed)
(Suppose this repository is already put the ideal directory, and unzipped. Suppose Python3 (>=3.12) is installed. If you would like to create a new python environment and install python 3.12 with conda, you can use 
```
conda create -n myenv python=3.12
```
and once the environment is created, use
```
conda activate myenv
```
to source the python environment.
)

### Install requirement packages using
```
pip3 install -r requirements.txt
```

### Run a test
Run
```
python3 solve.py spring_test
```
and then a folder `results` will be created with structure shown below. 

```
results/
├── log
│   └── Spring
│       └── test_0.txt
├── mdl
│   └── Spring
│       ├── nn_test_0_1000
│       ├── ... 
│       ├── nn_test_0_900
│       ├── optim_test_0_1000.pt
│       ├── ...
│       └── optim_test_0_900.pt
└── plot
    └── Spring
        ├── animation_test_0.gif
        ├── ...
        └── plot_test_0_00001001.png
```

Then configuration of this test run is in the file `./conf/spring_test.yaml`. Other config files are also in the `./conf/` directory.


### Run experiments in the paper
```
python3 run.py
```
You may modify the settings in the top lines of `run.py` to run experiments for specific problems or algorithms. 
