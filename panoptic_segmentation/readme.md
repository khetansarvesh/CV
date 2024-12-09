# Panoptic Segmentation


# ************** Problem Statement **************
This is a multitask problem wherein we do all of these tasks together i.e. multiple key point detection + object detection + Instance segmentation. 

![alt text](https://github.com/khetansarvesh/CV/blob/main/panoptic_segmentation/prb_statement.png)
In this above image we are not doing multiple key point detection but we can do that also

# ************** Solutions **************
Read [2 stage model from Object localization task](https://pub.towardsai.net/computer-vision-object-localization-task-2d536238510f) before doing this, it is on the similar lines


### Method 1 : 


### Method 2 : Region Based CNN (RCNN) Method
- Now instead of looping over all the possible sliding windows we heuristically calculate potential regions and then pass only these regions to the model !!
- Faster RCNN

### Method 3 : Region Proposal Network (RPN) Based Method
- Here the heuristic model is a DL based model
- Faster RPN (this method they called MASK REGION BASED CNN i.e. MASK R-CNN) method for this task was released in 2017 but is outdated now

### Method 4 : Detectron (2018) (outdated)

### Method 5 : Detectron 2 (2019) (outdated)


