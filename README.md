
# AIM Gait Anaylsis Toolkit
This is a tool kit for Gait Analysis to work with the Vicon tool kit.
It provides tools to analyze gaits, including separating the gait cycle

## Authors
- [Nathaniel Goldfarb](https://github.com/nag92) (nagoldfarb@wpi.edu)
- [Alek Lewis](https://github.com/ajlewis02) (ajlewis@wpi.edu)





## Dependencies 
This package requires:

* Python 3.8
* [AIM_Vicon](https://github.com/WPI-AIM/AIM_Vicon)
* [AIM_GaitCore](https://github.com/WPI-AIM/AIM_GaitCore.git)
* numpy
* scipy
* matplotlib
* pandas
* dtw




## Install
This package and its dependencies can be installed automatically via pip.

````bash
pip install git+https://github.com/WPI-AIM/AIM_GaitAnalysisToolkit.git
````
(If you have both Python 2 and Python 3 installed you'll need to use `pip3` - `pip` defaults to Python 2 if installed.)

## Upgrading
This package can be upgraded via pip:

```bash
pip install --upgrade git+https://github.com/WPI-AIM/AIM_GaitAnalysisToolkit.git
```

If necessary, Vicon and GaitCore can also be upgraded via pip:
```bash
pip install --upgrade git+https://github.com/WPI-AIM/AIM_Vicon.git
```
```bash
pip install --upgrade git+https://github.com/WPI-AIM/AIM_GaitCore.git
```

## Data
Gaiting data can be found here. 
* [AIM_GaitData](https://github.com/WPI-AIM/AIM_GaitData.git)


## Learning tools
* The Model folder holds the different algorthum to for training. 
It should extend ```ModelBase.py```. 
* The Trainer folder holds the trainer for the different algorithums. 
Its name should match the training algorithum used internial followed by Trainer. 
Follow the convention set. It should extend ```TrainerBase.py```.  
This class sould prepare the data to be trained and save the trained model.
* The Runner folder holds all the Runners for the trained model. It runs a trained model. 
It should follow the set naming convetion and extend ```RunnerBase.py```.   

