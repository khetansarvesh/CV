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

            # calculating (x_center,y_center) coordinates and (width, height) of the bounding box wrt the grid cell
            x_cell = self.S * x - int(self.S * x)
            y_cell = self.S * y - int(self.S * y)
            width_cell = width * self.S # width_pixels = (width*self.image_width), cell_pixels = (self.image_width), width_relative_to_cell = width_pixels/cell_pixels
            height_cell = height * self.S

            # If no object already found for specific cell i,j => Note: This means we restrict to ONE object per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell]) #bounding box coordinates
                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
