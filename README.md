# Computer Vision (CV)

# Digital Image Preprocessing

Digital image preprocessing involves manipulating input images to produce modified output images, enhancing their suitability for computer vision applications. This process aims to reveal hidden details or highlight specific areas of interest, resulting in a more suitable feature set for analysis.

## Understanding Images

Images are essentially matrices of numbers, representing:
- RGB color channels
- Binary data (0s and 1s)

## Dimensionality Reduction

To compress images and reduce their dimensions, we can employ unsupervised learning techniques:
- Principal Component Analysis (PCA) is often the most effective method

## Feature Extraction

Converting non-numerical data into numerical datasets requires domain expertise to identify relevant features that accurately define the image for the specific problem statement.

## Image Enhancement Techniques

### Normalizing Image
- Convert pixel values from 0-255 to 0-1 range

### Illumination Correction
- Various methods available, choose based on project requirements and previous research

### Handling Noise

### Duplicate Image Removal
- Check for exact and near-duplicate images

### Image Quality Control
- Remove blurry images
- Remove images with unusual aspect ratios
- Resize images with odd dimensions
- Remove low-information images

## Representation Learning / Tokenization

- Non-Modeling Based Methods
- Modeling Based Methods

## Downstream Computer Vision Tasks

### Non-Generative Tasks (Supervised CV)

1. Object Localization / Image Classification + Localization
2. Object Detection / Recognition
3. Semantic Segmentation
4. Instance Segmentation / Simultaneous Detection and Segmentation (SDS)
   - Advanced form of object detection with precise object outlines

### Generative Tasks (Unsupervised CV)

1. Style Transfer

2. Depth Estimation
   - Challenging task: understanding 3D structure from 2D images
   
   **Non-Learning Based Method:**
   - Stereo vision (outdated)
   - Requires two-camera setup (stereo pair)
   
   **Learning-Based Method:**
   - State-of-the-art approach
   - Monocular Depth Estimation (MDE) using a single camera
   - Notable models:
     - MiDas Model (2019)
     - Depth Anything Model V1 (Early 2024)
     - Depth Anything Model V2 (Late 2024, SOTA)

## Datasets

For a comprehensive list of datasets, refer to relevant computer vision resources and repositories.
