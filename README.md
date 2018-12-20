LV Data Map
================

These scripts provide a method to fuse information from multipe imaging sequences to build a patient-specific finite-element (FE) model of the left ventricle (LV).

## Introduction
This project provides a generic pipeline for mapping imaging data onto a LV FE mesh. Specifically, there are two scripts that follow the same pipeline to project scar and mechanical activation information onto a FE mesh. 

## Data & Scripts
The data files needed to run the example scripts are uploaded on [SimTK.](https://simtk.org/projects/lvdatamap)

The scripts for mapping are hosted here on github.

## MATLAB
The MATLAB data fusion routine uses the built-in [scatteredInterpolant](https://www.mathworks.com/help/matlab/ref/scatteredinterpolant.html) function to interpolate data from imaging across the LV.

- *LVMAP_Shell.m* projects scar from LGE MRI onto the LV, FE mesh
- *LVMAP_Shell_DENSE.m* projects mechanical activation from DENSE MRI onto the LV, FE mesh 

## Python
The Python (3.0+) data fusion routine uses the [scipy radial basis function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html) to interpolate data from imaging across the LV.

## Contact
For any questions regarding the library you can contact TK at phung@virginia.edu
