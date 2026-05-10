"""Optional editable install: ``pip install -e .``"""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="medprefix",
    version="1.0.0",
    description="Med-Prefix: Tri-Modal Prefix Conditioning for Dermatology Report Generation",
    author="Tran Thai Ha, Bui Thanh Hung",
    author_email="ha20040204@gmail.com",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["scripts", "configs", "data*", "results*"]),
    install_requires=requirements,
)
