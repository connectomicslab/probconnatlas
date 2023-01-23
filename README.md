## Description
This repository complements the paper submitted to __Scientific Data__ named _**A multi-scale probabilistic atlas of the human connectome**_.
The atlas and the set of open-source tools allow go beyond the description of a few well-known fiber bundles and allow to perform specific connectomic analysis with almost any brain imaging data (without the need for diffusion MRI per se). This should allow the broadest community of basic and clinical neuroscientists to perform high-end connectomice research. 
This atlas is derived from a cohort of 66 normal adult subjects from the Human Connectome Project (_[HCP]_), whose white matter (_WM_) connectivity has been investigated with the most up to date methods and in a multi-scale manner. This is a subsample of the original 100 unrelated subjects dataset freely available at the HCP website. Only the 70 subjects belonging to the releases Q1, Q2 and Q3 were included. From these subjects, the thalamic clustering failed to provide the expected segmentation pattern for three subjects and the registration of the streamlines to MNI space failed to provide the expected spatial alignment for one subject. Thus a final cohort of 66 healthy subjects was finally used in the atlas construction.
The IDs, age and gender of these subjects are provided in the CSV file called __subjects_66_HCP.csv__. An example of the data stored in this file is shown in the following table:

| Subject | Release | Acquisition | Gender | Age |
| ------ | ------ | ------ | ------ | ------ |
| 100307 | Q1 | Q01 | F | 26-30 |
| 100408 | Q3 | Q03 | M	| 31-35 |
| 101915 | Q3 | Q04 | F | 31-35 |

The atlas is referenced in standard [MNI] (_Montreal Neurological Institute_) space with a high resolution T1 weighted image (__ICBM 2009c Nonlinear Asymmetric__ ). Accordingly, registration of individual brain images of various imaging modalities can be matched with the atlas, and individual expected connectivity can be mapped and put in relation with individual imaging features.

## Connectome atlas
The developed multi-scale atlas is presented in four different files stored in **Hierarchical Data Format** ([HDF5] files with _**.h5**_ extension). Each file contains the probabilistic connectome atlas for each of the four scales. Each of these [HDF5] files contains the same groups and datasets. The main difference among them is the amount of data stored because it proportionally depends on the number of gray matter (_GM_) regions included in each parcellation scale. Each [HDF5] file contains three different groups of datasets: 1) **header**, 2) **matrices** and 3) **atlas**.

Inside the **header** group, the number of subjects employed to build the atlas and the required data to pass from the [HDF5] format to [Nifti-1] file format is contained in different datasets. This group also contains scale-specific information about the gray matter regions employed to separate the bundles.
These files can be downloaded from https://doi.org/10.5281/zenodo.4919131.

| Group/Dataset | Description |
| ------ | ------ |
| `header/nsubjects` | Number of subjects employed to build the atlas.|
| `header/dim` | Image dimensions.|
| `header/voxsize` | Voxel dimensions.|
| `header/affine` | Position of the image array data in MNI space.|
| `header/gmcodes`| Region codes (position in the matrix). |
| `header/gmregions`| Region names. |
| `header/gmcolors` | Region RGB (red, green and blue) colors triplets. It can be used for networks visualization. |
| `header/gmcoords` | Coordinates of the center-of mass in mm (MNI space). |

This information is useful for visualization purposes (ie. network visualization) and it is key to establish the relationship between the WM bundles and the real brain anatomy.

The **matrices** group contains three relevant connectivity matrices computed from the subjects sample used to create the multi-scale atlas:
 Group/Dataset | Description |
| ------ | ------ |
| `matrices/consistency` | Number of subjects in which at least one streamline is present connecting each pair of gray matter regions).|
| `matrices/numbStlines` | Average number of streamlines connecting each pair of gray matter regions.|
| `matrices/length` | Mean length of the streamlines connecting each pair of gray matter regions.|

Finally, the **atlas** group contains the coordinates and subject consistency for each of the voxels belonging to each WM bundle of the developed atlas. This group contains as many datasets as the number of scale-specific bundles. The names of the datasets are defined by the codes of the GM regions connected by the probabilistic bundle (e.g., **1_10** and **10_57**: bundles connecting the regions 1 and 10 and 10 and 57, respectively).
Each dataset contains a Nx4 matrix where N is the number of voxels belonging to the bundle. The first three columns are the X, Y and Z voxel coordinates in MNI template space, and the fourth column is the number of subjects where this specific voxel contains at least one streamline passing through it.

| Group/Dataset | Description |
| ------ | ------ |
| `atlas/1_10` | Nx4 matrix. Where N is the number of voxels belonging to the connection bundle **1_10**. |
| `atlas/10_57`| Nx4 matrix. Where N is the number of voxels belonging to the connection bundle **10_57**. |
| `altas/82_92`| Nx4 matrix. Where N is the number of voxels belonging to the connection bundle **82_92**. |

## Other files in the repository

1. Color-coded Nifti-1 images are also provided inside the __.zip__ file named __colored_wmbundles.zip__. There is one 4D volumetric Nifti-1 image for each scale where the three volumes along the fourth dimension represent the red, green and blue channels. Different colors are for different white matter bundles and the intensity of the colors are given by the probability of the voxel to belong to certain bundles. 

2. The average number of streamlines and the number of bundles passing through each voxel for each scale are stored in the compressed file called __bundcount_and_tdi.zip__. 

3. A set of tables (__sample_tables.zip__) with the information about the cohorts employed to build and evaluate the developed multi-scale connectome atlas.

## Acknowledgments

The used HCP data is provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

## Open-source tools to manipulate and apply the multi-scale connectome atlas

Here, a set of Python-based tools to apply the atlas within different research scopes are provided. 
The __wm_bundles_atlas.py__ is the main tool for manipulating the connectome atlas developed at Lausanne University Hospital. It is implemented as a Python 3 application running on Ubuntu Linux, relying on different Python packages such as __[h5py]__.

```sh
    $ python wm_bundles_atlas.py -s <ScaleId> -m <Method>  -p <ScalarMap> -o <OutputBasename> 
```

##### Options:
Brief description of input options:

| Option | Description |
| ------ | ------ |
| `--method`, `-m` | Method selection (**connectivity** or **consfile**).|
| `--out`, `-o` | Output basename. |
| `--scale`, `-s` | Scale Id (**scale1**, **scale2**, **scale3** or **scale4**) |
| `--bfile`, `-bf` | Nifti-1 image containing Regions of Interest or Comma-Separated Values (_**.csv**_) file. |
| `--map`, `-p` | Scalar map. |
| `--subth`, `-st` | Subject-level consistency threshold in percentage (0-100). |
| `--voxth`, `-vt` | Voxel-level consistency threshold. Probability values (0-1). |
| `--extract`, `-e` | Save individual bundles in Nifti-1 format. |
| `--all`, `-a` | Save individual bundles intercepting any of the regions of interest inside the mask. |
| `--collapse`, `-c` | Collapse the selected bundles into a 4D color-coded Nifti-1 file. |
| `--force`, `-f` | Overwrite the results. |
| `--verbose`, `-v` | Verbose (**0**, **1** or **2**). |
| `--help`, `-h` | Help. |

***

## Installation

Required python packages: 
- [h5py],  [numpy], [nibabel], [time], [os], [pathlib], [argparse], [sys], [csv]

---
## Usage  of the tool

#### 1. Computing connectivity matrices
Compute the mean value of a supplied scalar map along all the bundles included in the atlas for a specific scale. It outputs a connectivity matrix where the connection strength is the mean value of the scalar map along each bundle.

##### Examples
Compute the mean, median and standard deviation values of a supplied scalar map along all the bundles included in the atlas. It outputs a connectivity matrix where the connection strength is the mean value of the scalar map along each bundle. Another file should be supplied if connectivity is selected as method. This file should be supplied through the flags --map -p.
```sh
    $ python wm_bundles_atlas.py -s <ScaleId> -m connectivity -p <ScalarMap> -o <OutputBasename> 
```
Only the connections that appears in the 70% of the subjects will be selected.
```sh
    $ python wm_bundles_atlas.py -s <ScaleId> -m connectivity -p <ScalarMap> -o <OutputBasename> --subth 30
```
Only the connections that appears in the 73% of the subjects will be selected. For these connections, only the voxels appearing in the 80% of the subjects will be used.
```sh
    $ python wm_bundles_atlas.py -s <ScaleId> -m connectivity -p <ScalarMap> -o <OutputBasename> --subth 27 --voxth 0.2
```

##### Results
The resulting connectivity matrices are stored using the **Hierarchical Data Format** ([HDF5]).

These files are saved with extension _**.h5**_. 
:   **Mean connectivity matrix** (_-connmat-mean-scaleN.h5_)**:** The weights of this connectivity matrix represent the mean value of the scalar map along all the voxels belonging to the bundle connecting each pair of regions.
:   **Median connectivity matrix** (_-connmat-median-scaleN.h5_)**:** The connection strength is the median value of the scalar map including all the voxels belonging to the bundle connecting each pair of regions.
:   **Std connectivity matrix** (_-connmat-std-scaleN.h5_)**:** The connection strength is the standard deviation value of the scalar map including all the voxels belonging to the bundle connecting each pair of regions.

 
Only one group is stored inside each  HDF5 file (_**.h5**_) . This group contains different datasets with the scale-specific information about the gray matter regions employed to separate the bundles. This information is key to establish the relationship between the WM bundles and the real brain anatomy.

| Group/Dataset | Description |
| ------ | ------ |
| `connmat/matrix` | Connectivity matrix.|
| `connmat/gmcodes`| Region codes (position in the matrix). |
| `connmat/gmregions`| Region names. |
| `connmat/gmcolors` | Region RGB (red, green and blue) colors triplets. It can be used for networks visualization. |
| `connmat/gmcoords` | Coordinates of the center-of mass in mm (MNI space). |

These files can be loaded very easily using python.
```python
# Importing libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Reading the h5 file
connFilename = '<my-connmat-scaleN.h5>'
hf = h5py.File(hfName, 'r')

# Loading the matrix
tempVar       = hf.get('connmat/matrix')

# Ploting the matrix using matplotlib
plt.imshow(np.array(tempVar))
plt.colorbar()
plt.show()
```



#### 2. Selecting specific bundles

Extract and/or save some specific bundles. The desired bundles should be supplied through a Nifti-1 file or a Comma-Separated Value (_**.csv**_) text file using the flags `--bfile` or `-bf`. If a scalar map is supplied using the options `--map` or `-o`, a table with the mean values along these bundles is stored as well. The images will be saved only if the flags `--extract` or `--e` are specified. 
```sh
    $ python wm_bundles_atlas.py -s <ScaleId> -m consfile -p <ScalarMap> -o <OutputBasename> 
```

##### I. Using a mask file with one or more regions of interest 
If the file is a Nifti-1 image, the bundles intercepting its non-zero values will be extracted and/or saved. If the image contains different Region of Interest (_ROIs_) only the bundles connecting two or more ROIs will be selected. 
```sh
   $ python wm_bundles_atlas.py -s <ScaleId> -m consfile -p <ScalarMap> -o <OutputBasename> -bf <Mask>
```
If the flags `--all` or `-a` is specified then all the bundles intercepting any of these ROIs will be selected.
```sh
    $ python wm_bundles_atlas.py -s <ScaleId> -m consfile -p <ScalarMap> -o <OutputBasename> -bf <Mask> -a
```
If the flags `--collapse` or `-c` is specified then all the bundles will be collapses into a single 4D color-coded Nifti-1 file. It can be used to compute a binary mask of the bundles reaching the ROIs.
```sh
    $ python wm_bundles_atlas.py -s <ScaleId> -m consfile -p <ScalarMap> -o <OutputBasename> -bf <Mask> -c
```

> ##### II. Using a _*.csv_ file for one or more bundles 
If the file is a readable text file, each row of the txt file should contain the source and target regions separated by a comma (ie. 10,57). 
```sh
    $ python wm_bundles_atlas.py -s <ScaleId> -m consfile -p <Scalar Map> -o <Output Basename> -bf <CSV file> -c
```


###### Examples of  _*.csv_ text file
If the _**.csv**_ file contains two rows then the bundles between the source and the target regions codes will be selected. The first row of the file will be ignored because it is assumed that it contains the column headers. 
1. Bundles between two ROIs.

| SourceROI | TargetROI |
| ------ | ------ |
| 10| 57|
| 22| 48|
| 112| 28|
| 11| 45|

If the _**.csv**_ file contains only one row then the bundles reaching any of the specified regions will be selected. 
2. Bundles reaching one ROI.

| SourceROI | 
| ------ |
| 10|
| 22|
| 112|
| 11|

>  Note: Any of the options described for the usage of the mask can be used for the _**.csv**_ file

##### III. Results 
The main output result, if `--method` or `--m` flag is set to **consfile**, is a Comma-Separated Value file. This file stores a table containing different columns.

Columns meanings:
:   **BundleId:**  Codes of the two gray matter regions connected by this bundle.
:   **SourceROI** and **TargetROI:** Gray matter regions names.
:   **Consistency:**  Percentage of the subjects used to build the atlas that contained the bundle.
:   **MeanValue:**  Mean value of the scalar map along the bundle.
:   **StdValue:**  Standard deviation value of the scalar map along the bundle.
:   **MedianValue:**  Median value of the scalar map along the bundle.

_**Table example**_
| BundleId | SourceROI | TargetROI | Consistency | MeanValue | StdValue | MedianValue | 
| ------ | ------ | ------ | ------ | ------ | ------ |------ |
| 8_55 | ctx-rh-superiorfrontal | ctx-lh-superiorfrontal | 100.0 | 0.37 | 0.17 | 0.38 | 
| 9_56 | ctx-rh-caudalmiddlefrontal | ctx-lh-caudalmiddlefrontal | 98.48 | 0.39 | 0.15 | 0.39 | 
| 10_57 | ctx-rh-precentral | ctx-lh-precentral | 98.48 | 0.43 | 0.13 | 0.42 | 
| 11_55 | ctx-rh-paracentral | ctx-lh-superiorfrontal | 100.0 | 0.52 | 0.12 | 0.52 | 

If the options `--extract` or `-e` are specified, the individual bundles in Nifti-1 format will be saved. A 4D color-coded Nifti-1 file could be saved if the `--collapse` or `-c` is selected.

## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


   [HDF5]: <https://www.hdfgroup.org/solutions/hdf5/>
   [https://doi.org/10.5281/zenodo.4919131]: <https://doi.org/10.5281/zenodo.4919131>
   [Nifti-1]: <https://www.nitrc.org/docman/view.php/26/204/TheNIfTI1Format2004.pdf>
   [HCP]: <https://www.humanconnectome.org>
   [HCP website]: <https://db.humanconnectome.org/>
   [MNI]: <https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009> 
   [h5py]: <https://www.h5py.org/>
   [numpy]:<https://numpy.org/>
   [nibabel]:<https://nipy.org/nibabel/>
   [time]:<https://docs.python.org/3/library/time.html>
   [os]:<https://docs.python.org/3/library/os.html>
   [pathlib]:<https://docs.python.org/3/library/pathlib.html>
   [argparse]:<https://docs.python.org/3/library/argparse.html>
   [sys]:<https://docs.python.org/3/library/sys.html>
   [csv]:<https://docs.python.org/3/library/csv.html>
   
 
