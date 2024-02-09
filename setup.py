from setuptools import find_packages, setup

setup(
    author="Fariba Karimi",
    author_email="karimi@itis.swiss",
    version="1.1.0",
    description="Fourier Base Fitting on Masked Data",
    name="mift",
    packages=find_packages(where="src", include=["mift", "mift.*"]),
    package_dir={"": "src"},
    install_requires=["numpy", "matplotlib", "jax", "jaxlib", "ipykernel"],
)
