import setuptools

setuptools.setup(
    name="GaitAnalysisToolkit",
    version="1.0",
    install_requires=[
        "GaitCore @ git+https://github.com/nag92/AIM_GaitCore.git@1-fix-package-namespace",
        "Vicon @ https://github.com/nag92/AIM_Vicon.git@1-organize-modules",
        "pandas",
        "numpy",
        "scipy",
        "dtw",
        "matplotlib"
    ],
    packages=setuptools.find_packages(exclude=["testing", "Examples"])
)
