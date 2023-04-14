import random
import torch
import torch.nn as nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # coordinate loss
        self.mse = nn.MSELoss()
        # objectness loss
        self.bce = nn.BCEWithLogitsLoss()
        # class loss
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        """
        compute loss for single scale
        """
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i
         
        # no object loss
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # object loss 
        # anchor dim = (3, 2) (3 scale, (w, h)) -> (1, 3, 1, 1, 2)
        anchors = anchors.reshape(1, 3, 1, 1, 2) 
        box_preds = torch.cat(
            # x, y, w, h, 
            # x = sigmoid(t_x), y = sigmoid(t_y)
            # prediction box width = p_w * exp(t_w), prediction box higth = p_h * exp(t_h)
            [self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1
        )
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # not calculate the loss of binary classification of the object
        # but consider the iou of the predicted box and anchor box as the loss of the object
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))

        # box loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height of target box
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # class loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

        return loss