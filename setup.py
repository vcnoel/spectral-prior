from setuptools import setup, find_packages

setup(
    name="spectral_prior",
    version="0.1.0",
    description="Spectral Priors for Tabular Data Generation",
    author="Anonymous",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.10.0",
        "numpy<2.0.0",
    ],
    python_requires=">=3.8",
)
