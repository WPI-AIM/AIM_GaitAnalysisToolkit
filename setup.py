import setuptools

setuptools.setup(
    name="GaitAnalysisToolkit",
    version="1.0",
    install_requires=[
        "GaitCore @ git+https://github.com/WPI_AIM/AIM_GaitCore.git",
        "Vicon @ git+https://github.com/WPI_AIM/AIM_Vicon.git"
    ],
    packages=[
        "EMG",
        "LearningTools.Models",
        "LearningTools.Runner",
        "LearningTools.Trainer"
        "Model",
        "Session",
        "Trajectories"
    ]
)
