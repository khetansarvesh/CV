# OCR


# ************** Problem Statement **************
Identify the texts present in the image, as shown in below diagram.
![alt text](https://github.com/khetansarvesh/CV/blob/main/ocr/prb_statement.png)


# ************** Solution **************

## Step 1 : Text Detection => Identify the regions in the image which have text using the object detection model
![alt text](https://github.com/khetansarvesh/CV/blob/main/ocr/txt_det.png)

## Step 2 : Character Segmentation => Now for each of the text region, you need to segment out the character in that text
![alt text](https://github.com/khetansarvesh/CV/blob/main/ocr/char_seg.png)


## Step 3 : Character Classification => Now for each segmented character you need to run a classification model
![alt text](https://github.com/khetansarvesh/CV/blob/main/ocr/char_class.png)

