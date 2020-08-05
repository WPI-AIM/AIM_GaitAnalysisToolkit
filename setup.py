import setuptools

setuptools.setup(
    name="GaitAnalysisToolkit",
    version="1.1.1",
    install_requires=[
        "GaitCore @ git+https://github.com/WPI-AIM/AIM_GaitCore.git",
        "Vicon @ git+https://github.com/WPI-AIM/AIM_Vicon.git",
        "pandas",
        "numpy",
        "scipy",
        "dtw",
        "matplotlib"
    ],
    packages=setuptools.find_packages(exclude=["testing", "Examples"])
)
