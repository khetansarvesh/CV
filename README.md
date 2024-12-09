# ********** Digital Image Preprocessing **********

Here our aim is to enhance the image quality in the image dataset that we are dealing with. For a comprehensive list of datasets, refer to this [linkedin post](https://www.linkedin.com/feed/update/urn:li:activity:7269031439713656832/) by me. Following are few approaches that can be used :  

1. Normalizing Image : Convert pixel values from 0-255 to 0-1 range
2. Illumination Correction : Various methods available, choose based on project requirements and previous research
3. [Handling Noise](https://github.com/khetansarvesh/CV/blob/main/notes/Handling_Noise.pdf)
4. Duplicate Image Removal : Check if there are duplicate images or not (both exact and near duplicates) and remove if any
5. Remove blurry images
6. Remove images with unusual aspect ratios
7. Resize images with odd dimensions : Check if there are images with odd size, if yes then resize them
8. Remove low-information images












# ****** Representation Learning / Tokenization ******

- [Non-Modeling Based Methods](https://khetansarvesh.medium.com/non-modelling-methods-for-image-representation-learning-5a3b80a31a29)
- [Modeling Based Methods](https://www.google.com/url?q=https://khetansarvesh.medium.com/modelling-based-methods-for-image-representation-learning-48cd5f56ad0d&sa=D&source=docs&ust=1733758892238210&usg=AOvVaw1HeO_DRIuK6YLyDS-Ne3gT)










# ********** Downstream Computer Vision Tasks **********

### Non-Generative Tasks (Supervised CV)

1. [Object Localization / Image Classification + Localization Task](https://pub.towardsai.net/computer-vision-object-localization-task-2d536238510f)
2. [Object Detection / Recognition Task](https://pub.towardsai.net/computer-vision-object-detection-task-b572d58f2ada)
4. [Semantic Segmentation Task](https://levelup.gitconnected.com/semantic-segmentation-models-clearly-explained-780bde9515b5)
5. Instance Segmentation / Simultaneous Detection and Segmentation (SDS) Task : 
   - This is an advanced form of object detection task, in object detection task you just made bounding boxes but here you can make exact outline of the object
   - Check out this image [here](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*SNvD04dEFIDwNAqSXLQC_g.jpeg)
6. Single Human Key-Point Detection / Pose Estimation Task : Here the problem statement is to represent a person by K points. Refer this image [here](https://github.com/khetansarvesh/CV/blob/main/notes/pose_estimation_prb_statement.png).
7. Multiple Human Key Point Detection Task
8. Panoptic Segmentation Task
9. Photo Optical Camera Recognition (OCR) Task

### Generative Tasks (Unsupervised CV)

1. [Style Transfer](https://khetansarvesh.medium.com/image2image-translation-task-588d134e9ee8)

2. Depth Estimation
   - It is a difficult task because it requires the model to understand 3D structure using only 2D images. There are two ways to solve this.
   - **Non-Learning Based Method:** This was used earlier and not used anymore. This required 2 camera setup and hence called stereo pair. This method is called stereo vision.
   - **Learning-Based Method:** This method is SOTA and requires only 1 camera setup. Hence called monocular depth estimation (MDE). There have been many proposed models but the few best models are 
        - a. MiDas Model (2019)
        - b. [Depth Anything Model V1](https://depth-anything.github.io/) (Early 2024)
        - c. [Depth Anything Model V2](https://depth-anything-v2.github.io/) (Late 2024, SOTA)
