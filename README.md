This is a simple Python 3 code designed to make it a bit easier to visualise hypercubes.

Requirements:
- Python 3
- PyQt5
- Numpy
- Matplotlib

It has been noted that the ROIs can not display properly depending on the versions of the libraries. The following versions work:
Matplotlib version: 3.2.2
Numpy version: 1.22.0
Pandas version: 2.0.3
PyQt5 version: 5.15.11

To run this code, one must update lines 20, 21 and 22 (default wavelengths, reference spectra for the calibrated macbeth colour patches and the corresponding rgb values).
The code expects the hypercube and the wavelengths to be in the npz format.

<img width="1261" alt="image" src="https://github.com/YzaboRa/HypercubeVisualiser/assets/15233479/627b13c3-6d2a-4798-9db2-7f052b45c03a">
