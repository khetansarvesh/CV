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










# **********  Non-Generative Downstream CV Tasks (Supervised CV)  **********

1. [Object Classification](https://github.com/khetansarvesh/CV/tree/main/im_classification)
2. [Object Localization / Image Classification + Localization Task](https://pub.towardsai.net/computer-vision-object-localization-task-2d536238510f)
3. [Object Detection / Recognition Task](https://pub.towardsai.net/computer-vision-object-detection-task-b572d58f2ada)

4. [Semantic Segmentation Task](https://levelup.gitconnected.com/semantic-segmentation-models-clearly-explained-780bde9515b5)
5. Instance Segmentation / Simultaneous Detection and Segmentation (SDS) Task : 
   - This is an advanced form of object detection task, in object detection task you just made bounding boxes but here you can make exact outline of the object
   - Check out this image [here](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*SNvD04dEFIDwNAqSXLQC_g.jpeg)

6. Single Human Key-Point Detection / Pose Estimation Task : 
   - Problem Statement : Represent a person by K points. Most of the time K =14 cause almost everyone has 14 joints. Refer this image [here](https://github.com/khetansarvesh/CV/blob/main/notes/pose_estimation_prb_statement.png).
   - Solution Architecture : 
      1. This is very similar to Image Classification Architecture just that in the output layer you add 14*2 neurons representing coordinates of all these 14 joints. Refer architecture diagram [here](https://github.com/khetansarvesh/CV/blob/main/notes/pose_estimation_arch.png) 
      2. Instead of classification this is a regression problem hence loss function = Regression Loss = L2 Loss
   - If you change the dataset this same approach can also be used for other similar tasks like [Face Mesh Detection](https://ai.google.dev/static/edge/mediapipe/images/solutions/examples/face_landmark.png) / [Hand Detection](https://learnopencv.com/wp-content/uploads/2018/10/hand-output-skeleton.jpg) 
7. Multiple Human Key Point Detection Task : 
   - Same as above single human key point detection problem just that here we have multiple persons and for each person you need to do key point detection. 
   - Now during test time you will not know how many people will be there in the image. There can be 5 / 10 / 1000, we saw this similar kind of issue in case of object detection tasks.
   - Hence use the same strategy as used in the Object Detection task to solve this issue here.

8. [Panoptic Segmentation Task](https://github.com/khetansarvesh/CV/tree/main/panoptic_segmentation)
9. Photo Optical Camera Recognition (OCR) Task
    - Problem Statement : Identify the texts present in the image, as shown in image [here](https://github.com/khetansarvesh/CV/blob/main/notes/ocr_prb_statement.png)
    - Solution :
      
      a. Step 1 : Text Bounding Box Detection => Identify the regions in the image which have text using the object detection model. Refer image [here](https://github.com/khetansarvesh/CV/blob/main/notes/txt_det.png)
      
      b. Step 2 : Character Segmentation => Now for each of the text region, you need to segment out the character in that text. Refer image [here](https://github.com/khetansarvesh/CV/blob/main/notes/char_seg.png)

      c. Step 3 : Character Classification => Now for each segmented character you need to run a character classification model. Refer image [here](https://github.com/khetansarvesh/CV/blob/main/notes/char_class.png)
    





# **********  Generative Downstream CV Tasks (UnSupervised CV)  **********

1. Low Resolution Image to High Resolution Image

    a. [2016] SR GANs : [Paper Walkthrough](https://www.youtube.com/watch?v=fx-rXMcKlQc&list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va&index=14) || [Implementation](https://www.youtube.com/watch?v=7FO9qDOhRCc)
        
    b. [2017] Progressive GANs (ProGANs) : [Paper Walkthrough](https://www.youtube.com/watch?v=lhs78if-E7E) || [Implementation](https://www.youtube.com/watch?v=nkQHASviYac)
    
    c. [2018] Style GANs : [Video1](https://www.youtube.com/watch?v=qZuoU23ACTo) || [Video2](https://www.youtube.com/watch?v=TmN9SDqWKPk) || [Video3](https://www.youtube.com/watch?v=b3KGSsNSzZg)

    Similarly if you change the dataset then you can also do following tasks which these exact architectures
      - B/W Image to Colour Image
      - Blurry Image to Sharp Image
      - Day Image to Night Image
      - Summer Image to Winter Image
      - Noisy Image to Clean Image
      - ... 

2. [Style Transfer => Image2Image Translation](https://khetansarvesh.medium.com/image2image-translation-task-588d134e9ee8)
3. Depth Estimation
   - It is a difficult task because it requires the model to understand 3D structure using only 2D images. There are two ways to solve this.
   - **Non-Learning Based Method:** This was used earlier and not used anymore. This required 2 camera setup and hence called stereo pair. This method is called stereo vision.
   - **Learning-Based Method:** This method is SOTA and requires only 1 camera setup. Hence called monocular depth estimation (MDE). There have been many proposed models but the few best models are 
        - a. MiDas Model (2019)
        - b. [Depth Anything Model V1](https://depth-anything.github.io/) (Early 2024)
        - c. [Depth Anything Model V2](https://depth-anything-v2.github.io/) (Late 2024, SOTA)
