import torch
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        # loading images
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        # loading annotations
        ann_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        # for each image extracting class label and bounding box coordinates i.e. (x,y,width,height) from annotations file
        boxes = []
        with open(ann_path) as f:
            for ann in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in ann.replace("\n", "").split()]
                boxes.append([class_label, x, y, width, height])

        # converting these bounding boxes to tensors
        boxes = torch.tensor(boxes)

        # performing transformations on images and boxes
        transformations = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
        image, boxes = transformations(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # the above loading bounding box coordinates are relative to entire image now we are making it relative to grid
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # calculating which grid cell does the bouding box belongs to => (row, column)
            r = int(self.S * y)
            c = int(self.S * x)
            
            # calculating (x_center,y_center) coordinates and (width, height) of the bounding box wrt the grid cell
            x_cell = self.S * x - j
            y_cell = self.S * y - i
            width_cell = width * self.S #  = grid width / image width = width*self.image_width / self.image_width
            height_cell = height * self.S

            # Check if there is a bouding box assigned to this grid cell or not, if not then assign one and add the bounding box to it
            if label_matrix[i, j, 20] == 0:
                
                # 0 to 19 represents classes, and they are set to 0, setting the class that this belongs to as 1
                label_matrix[i, j, class_label] = 1 

                # 20 denotes it it has an object or not, so setting that here
                label_matrix[i, j, 20] = 1

                # 21 to 25 represents bounding box coordinates, setting that up
                label_matrix[i, j, 21:25] = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                # 26 to 29 is kept empty

        return image, label_matrix
