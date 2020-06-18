import setuptools

setuptools.setup(
    name="GaitAnalysisToolkit",
    version="1.0",
    install_requires=[
        "GaitCore @ git+https://github.com/WPI-AIM/AIM_GaitCore.git",
        "Vicon @ git+https://github.com/WPI-AIM/AIM_Vicon.git"
    ],
    packages=setuptools.find_packages(exclude=["testing", "Examples"])
)
