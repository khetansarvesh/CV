# Digital Image Preprocessing

In image processing our aim is to enhance the image quality

### 1. Normalizing Image
- Convert pixel values from 0-255 to 0-1 range

### 2. Illumination Correction
- Various methods available, choose based on project requirements and previous research

### 3. Handling Noise

### 4. Duplicate Image Removal
- Check for exact and near-duplicate images

### 5. Image Quality Control
- Remove blurry images
- Remove images with unusual aspect ratios
- Resize images with odd dimensions
- Remove low-information images











# Representation Learning / Tokenization

- [Non-Modeling Based Methods](https://khetansarvesh.medium.com/non-modelling-methods-for-image-representation-learning-5a3b80a31a29)
- [Modeling Based Methods](https://www.google.com/url?q=https://khetansarvesh.medium.com/modelling-based-methods-for-image-representation-learning-48cd5f56ad0d&sa=D&source=docs&ust=1733758892238210&usg=AOvVaw1HeO_DRIuK6YLyDS-Ne3gT)










# Downstream Computer Vision Tasks

### Non-Generative Tasks (Supervised CV)

1. [Object Localization / Image Classification + Localization Task](https://pub.towardsai.net/computer-vision-object-localization-task-2d536238510f)
2. [Object Detection / Recognition Task](https://pub.towardsai.net/computer-vision-object-detection-task-b572d58f2ada)
3. [Semantic Segmentation Task](https://levelup.gitconnected.com/semantic-segmentation-models-clearly-explained-780bde9515b5)
4. Instance Segmentation / Simultaneous Detection and Segmentation (SDS) Task : 
   - This is an advanced form of object detection task, in object detection task you just made bounding boxes but here you can make exact outline of the object
   - Check out this image [here](https://www.google.com/search?sca_esv=5430cf780e99fb8b&rlz=1C5CHFA_enIN1018IN1019&sxsrf=ADLYWILZsCVYnRmK_jyZ0RHfONdN0t5mqQ:1733755524781&q=instance+segmentation+task&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWd8nbOJfsBGGB5IQQO6L3J7pRxUp2pI1mXV9fBsfh39JqJxzRlphkmT2MeVSzs3MQEN5DgZmMeykT7pJra3boNMmeqljZ7P_Mci2RQNXWUvs0uYvk__uM-sA7iQHz0rC_qtHqiIQzRiax5tM1h82Y3dM9ZP1NkeLbOcAtw8KTZct2Q3sKq7AQMds098lJ3095meyIFw&sa=X&ved=2ahUKEwjQ6rnR9pqKAxVQ58kDHf6-FZoQtKgLegQIGRAB&biw=1680&bih=928&dpr=2#vhid=oIXOT5TiTWe2TM&vssid=mosaic) 

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










# Datasets

For a comprehensive list of datasets, refer to relevant computer vision resources and repositories.
