# OCR





# ************** Problem Statement **************
Identify the texts present in the image, as shown in below diagram.
![alt text](https://github.com/khetansarvesh/CV/blob/main/ocr/prb_statement.png)








# ************** Method 1 : **************

## Step 1 : Text Detection => Identify the regions in the image which have text using the object detection model
![alt text](https://github.com/khetansarvesh/CV/blob/main/ocr/txt_det.png)

## Step 2 : Character Segmentation => Now for each of the text region, you need to segment out the character in that text
![alt text](https://github.com/khetansarvesh/CV/blob/main/ocr/char_seg.png)


## Step 3 : Character Classification => Now for each segmented character you need to run a classification model
![alt text](https://github.com/khetansarvesh/CV/blob/main/ocr/char_class.png)









# ************** Method 2 : (more efficient) **************
Another idea is that since generating a letter is also dependent on the previous letter we can use sequence models too !! (these kind of architecture is good for OCR on bad images from early 17th century which have less manuscripts)

- Hence we will use CNN + RNN (together called CRNN) model for this process. Refer this [Medium Blog](https://towardsdatascience.com/handwriting-to-text-conversion-using-time-distributed-cnn-and-lstm-with-ctc-loss-function-a784dccc8ec3)
- In above architecture we can replace 
  a. CNN with a transformer 
  b. RNN with a transformer
  c. Transformer-RNN architecture / CNN-Transformer architecture / Transformer-Transformer architecture

- You can refer following resources for more informations
  a. [Paper 1](https://arxiv.org/pdf/2308.13418)
  b. [Paper 2](https://arxiv.org/pdf/2203.02378)
  c. [Blog](https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)

