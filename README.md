

# SPECTRA CALIBRATION


**Table of Contents**<a id='toc0_'></a> 

- [**Description of the Project**](#toc1_)
    - [**Task**](#toc1_1_)
    - [**Quick commands**](#toc1_2_)
    - [**Project directory overview**](#toc1_3_)
- [**Project tree**](#toc2_)
    - [**data_files**](#toc2_1_)
    - [**spectalpy**](#toc2_2_)
    - [**Makefile**](#toc2_3_)
    - [**script.py**](#toc2_4_)
        - [**Input**](#toc2_4_1)
        - [**Output**](#toc2_4_2)
- [**References**](#toc3_)
---

## <a id='toc1_'></a>[Description of the Project](#toc0_)

This project is an university exercise. 

> **Author:** [_Bernardo Vettori_](https://github.com/00-berni)
> 
> **Date:** 09/23/2023
> 
> **Version:** v23

### <a id='toc1_1_'></a>[Task](#toc0_)
The task is to make the calibration of the spectra taken during the 5 nights from the 26th to the 30th of July (2022) at the _Observatoire de Haute-Provence_ (OHP) in France.


This procedure is implemented by the modules in the `spectralpy` directory, that is the python package. To have more details see the section about [<u>the script</u>](#toc2_4_) and the package.

### <a id='toc1_2_'></a>[Quick commands](#toc0_)

- The list of used python packages and their versions is in the file `requirements.txt`. Hence, one have **to install them to run the script preventing possible errors of compatibility**.
    
    A quick way to do that is to use the command: `$ make requirements`.

- The fastest and safe procedure to run the script is to use the command: `$ make script`.

For other commands see the section about makefile.


### <a id='toc1_3_'></a>[Project directory overview](#toc0_)
In addition to the package (that is the aim of the exercise), the project directory contains:

- a directory to collect the acquired data
- a jupiter notebook to store and to explain in details every single step of the implementation (see the notebook section)
- a compilable file to do some operation quickly (see commands section or makefile section)
- a directory for tests (see the test section)

## <a id='toc2_'></a>[Project tree](#toc0_)

- [***data_files/***](data_files) - data directory

    - [***01_night/***](data_files/01_night) - data of the first night    
    - [***02_night/***](data_files/02_night) - data of the second night    
    - [***03_night/***](data_files/03_night) - data of the third night    
    - [***04_night/***](data_files/04_night) - data of the forth night    
    - [***05_night/***](data_files/05_night) - data of the fifth night
    - [***calibration/***](data_files/calibration) - documentation of the instruments
        - [`SpectreArNeLISA.pdf`](data_files/calibration/SpectreArNeLISA.pdf) : manual for the calibration of the _Alpy_'s lamp        
        - [`ThArV1_echelle.pdf`](data_files/calibration/ThArV1_echelle.pdf) : manual for the calibration of the _echelle_'s lamps        
    - [***chosen_echelle_data/***](data_files/chosen_echelle_data) - extracted data from acquisitions with _echelle_
    - [`objs_per_night.json`](data_files/objs_per_night.json) : list of all the observations
    
- [***notebook/***](notebook)

    - [`implementation_notebook.ipynb`](notebook/implementation_notebook.ipynb) : jupyter notebook

- [***results/***](results) - directory with the detected lines in calibrated spectra

- [***spectralpy/***](spectralpy) - the package

    - [`__init__.py`](spectralpy/__init__.py)
    - [`calcorr.py`](spectralpy/calcorr.py) : functions to compute the calibration of the _Alpy_'s data
    - [`data.py`](spectralpy/data.py) : functions to import data
    - [`display.py`](spectralpy/display.py) : functions to plot the data    

- [***tests/***](tests) - tests directory

- [`.gitignore`](.gitignore)

- [`LICENCE.md`](LICENCE.md) : the free licence

- [`Makefile`](Makefile) : compilable file for useful commands

- [`README.md`](README.md) : informations about the project     

- [`README.txt`](README.txt) : same file as this one in `.txt`

- [`requirements.txt`](requirements.txt) : require packages and versions

- [`script.py`](script.py)



### <a id='toc2_1'></a>[data_files](#toc0_)

All acquired data are listed in the file `objs_per_night.json`. The exstension of the files is the standard `.fit`. In addition to the observation data, for the chosen target the information about the cuts of the image is reported in ***calibration*** directory.

#### Observations 

Data of ***01_night*** and ***02_night*** are obtained thanks to the _Alpy_ spectroscope, while the others are taken with _echelle_. However, the routine can extrapolate only the _Alpy_'s data, then for some selected targets the extracted data from _echelle_ are just collected in the ***chosen_echelle_data*** directory[^1].

[^1]: The data from _echelle_ are extracted via a Matlab script written by Marco Monaci and used here with his permission.

- Alpy

    Each observation is acquired 
The spectrum of the lamp is acquired for each target. Data for calibration are taken manually.
A _flat_ was acquired for each night of observation.


### <a id='toc2_2'></a>[spectralpy](#toc0_)

The package is implemented to calibrate the acquired spectra and to extract only the data from the _Alpy_'s acquisitions. It is made up of 4 modules:

- `__init__.py`

    This module is imported automatically when the package is imported. It is used only to imported the module `data.py`.

- `calcorr.py`

    This is the main part of the project. The module collects all the functions to calibrate the spectra after data are extracted. In brief the procedure can be divided in 5 steps:

    1. **Select a target and extract its data**
    
    2. **Correct the inclination**
    
        Typically the slit is not aligned with the CCD and as a result the image inclination is not zero. To correct this effect the routine computes the inclination angle through a linear fit and then routates the image [[1]](#ref:not). The fit is obtained by studying the trend of the brithest pixel in each column varying with the x coordinate.
    
    3. **Compute the cumulative light**

        Each horizontal row of the image represents the same spectrum (the only difference is the brightness value), therefore the 2-D matrix is reduced to a 1-D array by summing the light on each column. 
    
    4. **Correct by _flat_**

        _Flat_ is reduced as explained above, too. However, before correcting the spectrum of the target, it is necessary to normalize the _flat_ and the normalization coefficient is simply the maximum value in brightness. Then the spectrum $\mathcal{S}_x$ becomes: 
        $$ \mathcal{S}_x \longrightarrow \frac{\mathcal{S}_x}{\mathcal{F}_x} $$
        where $\mathcal{F}_x$ is the normilized _flat_. The subscript $x$ means that the value of the function depends on the column of the image (its x coordinate).

    5. **Calibrate the spectrum**

        To calibrate the specrtum a reference is essential. The NeAr lamp is the standard lamp for the Alpy and the documentation about it is reported in the [`SpectreArNeLISA.pdf`](data_files/calibration/SpectreArNeLISA.pdf) file. After adjusting the inclination of the image the program computes the parameters of the function to pass from pixels (coordinates along x axis) to wavelengths (in $\AA$). To do that a polynomial fit with 3 parameters are used. The result of this procedure is the spectrum $\mathcal{S}_\lambda$.

The spectrum thus obtained is not normalized in brightness, because there are not enough observations of the same target to do it. 

After a spectrum has been calibrated, the last step of the routine changes a little: in fact, to avoid the samples of each calibration lamp the program computes the correlation between this and the previous lamp and compares it with the autocorrelation of the first. For all the investigated cases the difference is null.  


- `data.py`

    This module stores the paths of all directories used in the ruotine. It also provides functions to import, to read and to extract `.fit` data. The `get_data_fit()` function permits in addition to remove hot pixel from the image (via the `hotpx_remove()` of module `calcorr.py`) and to display it (via the `showfits()` of module `display.py`).

- `display.py`    

    This module provides a procedure to make plots quickly. It contains also the function to display `.fit` files or a chosen cut of the images.

### <a id='toc2_3_'></a>[Makefile](#toc0_)

### <a id='toc2_4_'></a>[script.py](#toc0_)

#### <a id='toc2_4_1_'></a>[Inputs](#toc0_)

The script takes :


#### <a id='toc2_4_2_'></a>[Outputs](#toc0_)



## <a id='toc3_'></a> [References](#toc0_)

1. <a id='ref:not'></a> Notes of _Astrofisica Osservativa_ course, 2021-2022.
