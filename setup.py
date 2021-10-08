from pathlib import Path
from setuptools import setup, find_packages

NAME = "iirnet"
DESCRIPTION = "IIRNet"
URL = "https://github.com/csteinmetz1/IIRNet"
EMAIL = "c.j.steinmetz@qmul.ac.uk"
AUTHOR = "Christian Steinmetz"
REQUIRES_PYTHON = ">=3.7.11"
VERSION = "0.0.1"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=["iirnet", "iirnet/baselines"],
    install_requires=[
        "torch",
        "torchaudio",
        "torchvision",
        "pytorch_lightning",
        "scipy",
        "matplotlib",
        "wget",
    ],
    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
)
