# Fourier Base Fitting on Masked Data
This python library is offering Fourier Base Fitting on structured data where part of the data is missing or unreliable. This is often the case in biomedical image data, for example the deformation data which shows how the brain pulsates during each cardiac cycle [[1]](https://www.sciencedirect.com/science/article/pii/S221315822200345X). 

## Description
The library was designed for data in 1D, 2D, and 3D (with one or multiple time steps like in the case of deformation data). For detailed information and mathematical description see [2]. 

## Installation
```
  pip install git+https://github.com/ITISFoundation/MIFT
```

## Usage
All the relevant functions are implemented in classes.py and utils.py.
To view examples of usage in 1D, 2D, and 3D space, please refer to the benchmarks located in the benchmarks folder for 1D, 2D, and 3D data, respectively.
For practical demonstrations of the library, please refer to the tutorials to observe how the library can be utilized for processing and reconstructing real deformation data. Sample deformation data can be accessed via: https://zenodo.org/records/10590047. 

## Support
If you have any questions or issues, contact karimi@itis.swiss. 

## Authors and acknowledgment
This library was prepared by Fariba Karimi under the direct guidance of Dr. Esra Neufeld and Dr. Arya Fallahi. 
This was developed for processing of the deformation data in the contect of developing a non-invasive surrogate for intracranial pressure monitoring (see [[1]](https://www.sciencedirect.com/science/article/pii/S221315822200345X) for detailed information). 

## License
GNU GENERAL PUBLIC LICENSE -- see full license text [here](LICENSE).

## References
[1] Karimi, F., Neufeld, E., Fallahi, A., Boraschi, A., Zwanenburg, J.J., Spiegelberg, A., Kurtcuoglu, V. and Kuster, N., 2023. Theory for a non-invasive diagnostic biomarker for craniospinal diseases. NeuroImage: Clinical, 37, p.103280. [Available here](https://www.sciencedirect.com/science/article/pii/S221315822200345X)

[2] Karimi, F., Neufeld, E., Fallahi, A., Kurtcuoglu, V. and Kuster, N., 2024. Efficient Fourier Base Fitting on Masked or Incomplete Structured Data. 
