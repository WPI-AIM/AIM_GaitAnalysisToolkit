
# AIM Gait Anaylsis Toolkit
This is a tool kit for Gait Analysis to work with the Vicon tool kit.
It provides tool to anaylsis gaiting including seperating the gait cycle

## Authors
- [Nathaniel Goldfarb](https://github.com/nag92) (nagoldfarb@wpi.edu)


## Dependence
* python 2.7
* numpy
* scipy
* matplotlib


## External Dependence 
All packages are installed in the `lib` folder

* [AIM_Vicon](https://github.com/WPI-AIM/AIM_Vicon)
* [AIM_GaitCore](https://github.com/WPI-AIM/AIM_GaitCore.git)




## Install
This package as level submoduls that need to be installed

````bash
git clone https://github.com/WPI-AIM/AIM_GaitAnalysisToolkit.git
cd AIM_GaitAnalysisToolkit
git submodule update --init --recursive
git submodule update --recursive