# Fluidized Bed CT

Reconstruction scripts for the TU Delft triple source-detector set-up. The 
scripts allow
 - a calibration of the set-up, using _Cate_, see _scripts/calib/_;
 - reconstruction of a static object, using a rotation table;
 - preprocessing and referencing;
 - reconstruction of a dynamic fluidized bed, using three angles per timestep.

## 1. Installation
For an installation with _conda_ (or _mamba_, preferred), run:
```shell
conda create -n fluidized_bed_ct python=3.10
conda activate fluidized_bed_ct
conda install numpy scipy imageio matplotlib joblib tqdm pyqtgraph -c conda-forge
conda install astra-toolbox -c astra-toolbox/label/dev
pip install transforms3d
```

Then download this package:
```shell
git clone https://github.com/adriaangraas/fbrct
```
Currently, the repositories do not contain a _setup.py_. To run a script, 
make sure that Python finds the modules by setting the _PYTHONPATH_ 
environment variable, e.g.:
```shell
conda activate fluidized_bed_ct
PYTHONPATH=path/to/fbrct python some_script.py
```
### Adapting for Windows
If you're running the scripts from windows, in `fbrct/loader.py`, the `PROJECTION_FILE_REGEX` needs to be adapted to `"camera ([1-3])\\\img_([0-9]{1,6})\.tif$"` to deal with backslash folder separation in Windows

## 2. Run a calibration

First install the _CaTE_ scripts are available to Python.

```shell
git clone https://github.com/adriaangraas/cate
```

In **scripts/calib/** there are scripts that show how to calibrate the
geometry using a marker object with glued metal markers on it. The scripts can
be modified to your needs.
1. First, using _ImageJ_ or an image editor of choice, find the first and 
   last frame of rotation. Use these as `proj_start` and `proj_end`
   values. Also select three frames for annotation. More is possible but not necessary.
   A good idea is to space them equally apart. For example, if the setup
   moves between frames 11 and 110 then take frames 11, 44, 77. Sometimes
   markers are barely visible in the column walls. That is fine, just make
   a guess, or pick a different frame. The frames don't have to be perfectly
   spaced. Use the chosen values in `t_annotated`.
2. The function `annotated_data()` helps quickly selecting marker points in the images. 
   The tool is a bit rudimental, but should get the job done. If `open_annotation`
   is `True` the tool annotates, if it is `False`, the tool returns previously
   annotated values. The function creates some NumPy files in the background
   to store the annotations.
3. The script then builds a triangular geometry and uses that as an initial
   guess for the set-up. The parametrization is that of a static set-up with
   three sources and detectors that undergoes fixed rotations along a single
   axis. The geometry with unknown parameters is stored in `multicam_geom`.
4. Then a function called `marker_optimization()` takes the geometries, and
   looks to find their parameters. This uses the _CaTE_ machinery to convert
   the geometry to a list of values that can be optimized with the nonlinear
   least-squares solver from SciPy. To find initial values for the markers,
   the annotated markers are found using a least-squares intersection in the
   3D volume. If `plot=True` is passed, a plot will show the positions of
   the markers with their lines of projection. This should show how good/bad 
   the current solution is.  The CaTE geometries are stored in a file `geom_{...}.npy` afterward. These
   files need to be copied and used for later reconstructions. The found
   marker positions are also stored, but only for later convenience.

The (optional) script _part02_test_recon.py_ is to help figure out 
out how accurate the reconstruction is. The projections from the marker scan be used to reconstruct
the marker object. There is a superfluous amount of data, since 3 detectors
are used for a rotational scan. To reconstruct the object, the geometry 
parameters of all not-annotated projections can be found by interpolation
in the direction of rotation. If the reconstructed object looks sharp, the
geometry parameters are likely correct. For a more detailed quantity, the
reconstructed marker column again be forward-projected again, giving
simulated projection images. The residual between the real projection
images, and the simulated projection images, i.e., y - AA^Tx, should be
small.
