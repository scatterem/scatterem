from setuptools import find_packages, setup

setup(
    name="scatterem2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "matplotlib",
        "pytorch-lightning",
        "kornia",
        "pyyaml",
        "scikit-image",
        # "tinycudann",  # Note: This may need special installation instructions
        "warp-lang",  # Note: This may need special installation instructions
    ],
    author="Philipp Pelz, Shengbo You, Nikita Palatkin",
    author_email="philipp.pelz@fau.de",
    description="A package for electron microscopy simulation and reconstruction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pelzphil/scatterem2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
