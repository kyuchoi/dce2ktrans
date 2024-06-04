# dce2ktrans
This is an official code for "Deep Learning Enhances the Reliability of Dynamic Contrast-Enhanced MRI in Diffuse Gliomas: Bypassing Post-processing and Providing Uncertainty Maps"

## Summary
A spatiotemporal deep learning model can improve the reliability of DCE-MRI by bypassing the estimation of the AIF and providing uncertainty maps, without diagnostic performance in diffuse glioma grading.

## Training
To be updated

## Inference
To be updated

## Dataset
The dataset was temporally split so that the test set (102 patients) consisted of scans taken after March 2016. The remaining 219 patients were randomly split into the training set (165 patients) and the validation set (62 patients).

### MRI Acquisition
- **Equipment**: 3T MRI units (Magnetom Verio or Magnetom Skyra, Siemens Healthineers) with a 32-channel head coil.
- **Protocol**: Glioma study including DCE-MRI, pre/post-contrast enhanced MP-RAGE T1WI, axial T2WI, and axial T2-FLAIR sequences.
- **DCE-MRI Parameters**: 
  - 3D T1-weighted spoiled gradient-echo sequence
  - TR/TE: 2.8/1.0 msec
  - Flip angle: 10°
  - Matrix: 192 × 192
  - FOV: 240 × 240 mm
  - Section thickness: 3 mm
  - Voxel size: 1.25 × 1.25 × 3 mm³
  - 60 acquisitions with 4-second temporal resolution
  - Gadobutrol administered (0.1 mmol/kg) with saline bolus
  - Total acquisition time: 5 minutes and 8 seconds
- **MP-RAGE Parameters**: 
  - TR/TE: 1370–1600/1.9–2.8 msec
  - Flip angle: 9°
  - Matrix: 256 × 232
  - FOV: 250 × 250 mm
  - Section thickness: 1 mm
- **Axial T2-WI Parameters**: 
  - TR/TE: 5100/89 msec
  - Flip angle: 150°
  - Matrix: 640 × 348
  - FOV: 199 × 220 mm
  - Section thickness: 5 mm
  - Number of excitations: 3
- **Axial FLAIR Parameters**: 
  - TR/TE: 8000–9000/90–97 msec
  - Inversion time: 2300–2500 msec
  - Flip angle: 130–150°
  - Matrix: 384 × 209–278
  - FOV: 199 × 220 mm
  - Section thickness: 5 mm
  - Number of excitations: 1–2

### Data Processing
- **Conversion**: All DICOM files converted to NIfTI gzipped files.
- **Corrections**: Motion and N4 bias field corrected.
- **DCE Image**: Time-averaged DCE image skull stripped and resampled to 256x256 resolution in the xy-plane.
- **Tumor Segmentation**: 
  - T1WI, T2WI, FLAIR images skull stripped using BET package.
  - Images centered, re-oriented (RAS+), and resampled to 1mm isotropic voxels.
  - T2WI and FLAIR images registered to T1WI space using ANTs SyN algorithm.
  - HD-GLIO neural network used for tumor segmentation.
  - Tumor segmentation maps registered to DCE space using rigid affine transform from ANTs package.