import cgitb
import config
import numpy as np
import os
import pandas as pd
import torch
import random
import glob
import csv

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    """
    This class is used to load the data from the csv file and the image directory.
    
    Parameters
    ----------
    csv_file : str
        The path to the csv file containing the image names and the label names.
    img_dir : str
        The path to the directory containing the images.
    label_dir : str
        The path to the directory containing the labels.
    anchors : list
        The list of anchors.
    image_size : int
        The size of the image.
    S : list
        The list of the number of cells in each scale.
    C : int
        The number of classes.
    """  
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):   
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # 9 anchors for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5 # used in non-max suppression, leave only the box with the highest confidence

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        # load label
        label_path = self.annotations.iloc[index, 1]
        # (class, center_x, center_y, width, height) ->  (center_x, center_y, width, height, class)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        # print(bboxes)
        # load image
        img_path = self.annotations.iloc[index, 0]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # 6 = (obj, x, y, w, h, class)
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        # loop over all the target bounding boxes
        for box in bboxes:
            # calculate iou between the box and the anchors with width and height only
            iou_anchors= iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            # make sure there is a bounding box responsible for each scale
            has_anchor = [False, False, False]
            # loop over the anchors, 9 anchors in total, 3 in sacale 1, 3 in scale 2, 3 in scale 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x) # the cell index
                # which anchor is responsible for the target bounding box
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                # if the anchor is not taken by any bounding box
                if not anchor_taken and not has_anchor[scale_idx]:
                    # there is an object in this cell
                    anchor_taken = 1
                    # the center of the bounding box
                    x_cell, y_cell = S * x - j, S * y - i
                    # the width and height of the bounding box, width is releative to the cell width
                    width_cell, height_cell = (
                        width * S, 
                        height * S,
                    )
                    # (x, y, w, h) are relative to the cell size and location
                    box_coordinate = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    # (obj, x, y, w, h, class)
                    targets[scale_idx][anchor_on_scale, i, j, :] = torch.tensor(
                        [1, *box_coordinate, int(class_label)]
                    )
                    has_anchor[scale_idx] = True
                # if the anchor is not taken by any bounding box but the scale is already taken by another bounding box
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # ignore the target bounding box
                    anchor_taken = -1
        
        return image, tuple(targets)

def test():
    dataset = YOLODataset(
        "defect_image/train.csv",
        "defect_image/images/",
        "defect_image/labels/",
        S=config.S,
        anchors=config.ANCHORS,
        transform=config.test_transforms,
    )
    S = config.S
    scaled_anchors = torch.tensor(config.ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(len(boxes))
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


def create_train_test_csv(image_path, label_path, ratio=config.SPLIT_RATIO):
    image_list = sorted(glob.glob(image_path+'/*'))
    label_list = sorted(glob.glob(label_path+'/*'))
    random.seed(1)
    random.shuffle(image_list)
    random.seed(1)
    random.shuffle(label_list)

    split_index = int(ratio * len(image_list))
    train_csv = (image_list[:split_index], label_list[:split_index])
    test_csv = (image_list[split_index:], label_list[split_index:])

    train_iterable = []
    test_iterable = []
    for i in zip(train_csv[0], train_csv[1]):
        train_iterable.append(i)
    for i in zip(test_csv[0], test_csv[1]):
        test_iterable.append(i)

    with open('defect_image/train.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(train_iterable)
    with open('defect_image/test.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(test_iterable)

if __name__ == "__main__":
    test()




