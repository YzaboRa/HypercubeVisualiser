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

<img width="1261" alt="image" src="https://github.com/user-attachments/assets/47666333-782e-4c56-9ccc-c8c927dbc1a8">

## To Do
- [ ] Add option to not display patch spectra (for clinical data)
- [ ] Add automatic detection and handling of different numbers of wavelengths (for ExVision data)

