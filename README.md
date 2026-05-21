# Fluidized Bed CT

Reconstruction scripts for the TU Delft triple source-detector set-up. The 
scripts allow
 - a calibration of the set-up, using `CaTE`, see `scripts/calib/`;
 - reconstruction of a static object, using a rotation table;
 - preprocessing and referencing;
 - reconstruction of a dynamic fluidized bed, using three angles per timestep;
 - reconstruction of a time-averaged fluidized bed / bubble column.

## 1. Installation
If you don't have _conda_ installed, find some installation instructions
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 
Do **not** use the TU Delft Software Center version of anaconda.

The fastest way of installing is to clone this repository and create a `conda`
environment from the `requirements.yaml` file in there. All explicitly installed 
packages have been exported there so there's a good chance conda can make a 
functional environment out of it.

### 1.1. Cloning
In the terminal, navigate to a folder where you want to keep the code
(preferable local, on the `C:` drive \[for Windows\]). There, download this
repository:
```shell
git clone https://github.com/RikVolger/fluidized-bed-ct.git
```
### 1.2. Create `conda` environment
You can then install the environment specified in requirements.yaml as a new 
environment:
```shell
cd path/to/fluidized-bed-ct
conda env create -f requirements.yaml
conda activate fluidized_bed_ct
conda develop .
```
> **_NOTE:_**  There's also the `environment.yaml` file. This is a full export 
> of the working environment on a specific computer that leaves no freedom to 
> `conda` when creating the environment. It's unlikely that this can be used to 
> create a functional environment on your machine, but it can be used to trace 
> down version-dependent errors.

### 1.3. Install _CaTE_ dependency
Now, install the _CaTE_ package:
```shell
cd path/to/fluidized-bed-ct
cd ../
git clone https://github.com/adriaangraas/cate
cd cate
conda activate fluidized_bed_ct
conda develop .
```

<!-- ### 1.x. Legacy install
Below the 'legacy' instructions for installation. It will potentially use newer
versions of packages, which might or might not work.
For an installation with _conda_ (or _mamba_, preferred), open a terminal window
and run:
```shell
conda create -n fluidized_bed_ct python=3.10
conda activate fluidized_bed_ct
conda install numpy scipy imageio matplotlib joblib tqdm pyqtgraph conda-build transforms3d tifffile pyvista -c conda-forge
conda install astra-toolbox -c astra-toolbox/label/dev
pip install transforms3d
```

In the terminal, navigate to the folder where you want to keep the code 
(preferably local, on the `C:` drive \[for Windows\]).
Then download this package:
```shell
git clone https://github.com/RikVolger/fluidized-bed-ct.git
```
Currently, the repositories do not contain a _setup.py_. To run a script 
(e.g. `some_script.py`), make sure that Python finds the modules by adding the 
folder to _PYTHONPATH_ e.g. through `conda develop`:
```shell
cd path/to/fluidized-bed-ct
conda activate fluidized_bed_ct
conda develop .
```

The calibration relies on the _CaTE_ scripts. Install these next to 
`fluidized-bed-ct`:

```shell
cd path/to/fluidized-bed-ct
cd ../
git clone https://github.com/adriaangraas/cate
cd cate
conda develop .
``` -->

## 2. Run a calibration

In **scripts/calib/** there are scripts that show how to calibrate the
geometry using a marker object with glued metal markers on it. The scripts can
be modified to your needs.
The calibration relies on the values indicated in `calib.yaml` in the root 
folder. Here, you indicate the root folder for the calibration files (e.g. 
`U:\XRay RPT ChemE\X-ray\Xray_data\2024-11-14 Rik en Sam`), and the folder
containing the actual images (e.g. `Rotation_needles_5degps_again`).
1. First, using _ImageJ_ or an image editor of choice, find the first and 
   last frame of rotation. Use these as `rotation:start` and `rotation:stop`
   values. _n_ frames for annotation are automatically selected, starting at
   `frames:start`. 3 frames is generally enough. More is possible but not
   necessary.
   Sometimes markers are barely visible in the column walls. That is fine, just
   make a guess.
2. The actual calibration is done in the script `part01_calibrate.py`. When you 
   run the script, a couple things happen:
   1. The function `annotated_data()` helps quickly selecting marker points 
      in the images. The tool is a bit rudimental, but should get the job done.
   2. If `open_annotation` is `True` the tool annotates, if it is `False`, the 
      tool returns previously annotated values. The function creates some NumPy 
      files in the background to store the annotations. These are stored next to 
      your raw data, in a folder `calib`.
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
      the current solution is.  The CaTE geometries are stored in a file
      `geom.npy` afterward, in the `calib` folder. The found marker positions 
      are also stored, but only for later convenience.

The (optional, but recommended) script _part02_test_recon.py_ is to help figure
out how accurate the reconstruction is. The projections from the marker scan be
used to reconstruct the marker object. There is a superfluous amount of data,
since 3 detectors are used for a rotational scan. To reconstruct the object, the
geometry parameters of all not-annotated projections can be found by
interpolation in the direction of rotation. If the reconstructed object 
(especially the needles) looks sharp, the geometry parameters are likely correct.

The third file, `part03_amend.py` serves some forgotten function and has not 
seen any use in a few years.

## 3. Reconstruct your measurements
The heart of the reconstructions is the file `scripts/reco_from_yaml.py`. This 
file uses a `scans_###.yaml` input file from the root directory, indicated in 
line 199. 

The file `scans_example.yaml` contains a number of examples of what the input 
could look like. 

The output of the reconstructions are .vtk files that can be used for 
visualization in ParaView (for fancy videos) or Python (for simpler images).