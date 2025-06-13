# Informed machine learning with a stochastic gradient-based algorithm

Under review as a conference paper at ICLR 2025, 
authors: Qi Wang (Lehigh University), Christian Piermarini (Sapienza University of Rome), Frank E. Curtis (Lehigh University).

### Install Python (Skip if Python3 (>=3.12) is already installed)
If you need to create a new Python environment and install Python 3.12 with conda, you can use the following command:
```
conda create -n myenv python=3.12
```
Once the environment is created, use:
```
conda activate myenv
```
to activate the Python environment.

### Install required packages
cd to this repository, e.g., `cd [your path]/SQPPIML`, and then install the required packages by
```
pip3 install -r requirements.txt
```

### Run a test
To run a test, use:
```
python3 solve.py spring_test
```
A folder `result_test` will be created with the structure shown below:

```
result_test/
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

The configuration for this test run is in the file `./conf/spring_test.yaml`. Other config files are also in the `./conf/` directory. The 'results' folder will 
be created within the SQPPIML folder.

### Run experiments in the paper
```
python3 run.py
```
You may modify the settings in the top lines of `run.py` to run experiments for a specific problem, algorithm, or learning rate.
The configurations for all the test problem are set with the settings fixed for the experiments
described in the paper. 
