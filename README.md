# ********** Digital Image Preprocessing **********

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











# ********** Representation Learning / Tokenization **********

- [Non-Modeling Based Methods](https://khetansarvesh.medium.com/non-modelling-methods-for-image-representation-learning-5a3b80a31a29)
- [Modeling Based Methods](https://www.google.com/url?q=https://khetansarvesh.medium.com/modelling-based-methods-for-image-representation-learning-48cd5f56ad0d&sa=D&source=docs&ust=1733758892238210&usg=AOvVaw1HeO_DRIuK6YLyDS-Ne3gT)










# ********** Downstream Computer Vision Tasks **********

### Non-Generative Tasks (Supervised CV)

1. [Object Localization / Image Classification + Localization Task](https://pub.towardsai.net/computer-vision-object-localization-task-2d536238510f)
2. [Object Detection / Recognition Task](https://pub.towardsai.net/computer-vision-object-detection-task-b572d58f2ada)
3. [Semantic Segmentation Task](https://levelup.gitconnected.com/semantic-segmentation-models-clearly-explained-780bde9515b5)
4. Instance Segmentation / Simultaneous Detection and Segmentation (SDS) Task : 
   - This is an advanced form of object detection task, in object detection task you just made bounding boxes but here you can make exact outline of the object
   - Check out this image [here](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*SNvD04dEFIDwNAqSXLQC_g.jpeg) 

### Generative Tasks (Unsupervised CV)

1. [Style Transfer](https://khetansarvesh.medium.com/image2image-translation-task-588d134e9ee8)

2. Depth Estimation
   - It is a difficult task because it requires the model to understand 3D structure using only 2D images. There are two ways to solve this.
   - **Non-Learning Based Method:** This was used earlier and not used anymore. This required 2 camera setup and hence called stereo pair. This method is called stereo vision.
   - **Learning-Based Method:** This method is SOTA and requires only 1 camera setup. Hence called monocular depth estimation (MDE). There have been many proposed models but the few best models are 
        - a. MiDas Model (2019)
        - b. [Depth Anything Model V1](https://depth-anything.github.io/) (Early 2024)
        - c. [Depth Anything Model V2](https://depth-anything-v2.github.io/) (Late 2024, SOTA)










# ********** Datasets **********

For a comprehensive list of datasets, refer to this [linkedin post](https://www.linkedin.com/feed/update/urn:li:activity:7269031439713656832/) by me.
