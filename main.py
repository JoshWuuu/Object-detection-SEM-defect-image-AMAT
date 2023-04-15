import config 
import os

import torch
import torch.optim as optim

from utils import (
    load_checkpoint,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
from model_build import YOLOv3
from model_train import train_fn
import config
from dataset import create_train_test_csv

def main():
    os.makedirs('tuned_model', exist_ok=True)
    # create train and test csv files, with image and label paths
    create_train_test_csv(config.IMG_DIR, config.LABEL_DIR)

    # create model, optimizer, train and test loaders
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    train_loader, test_loader, _ = get_loaders(
        train_csv_path=config.DATASET + "/train.csv",
        test_csv_path=config.DATAAZSET + "/test.csv",
    )

    # load model if specified
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )
    
    # create loss function and scaler
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    # scale anchors, is the anchor size multiplied by the grid size, relative to the cell size
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # train model
    train_fn(train_loader, test_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

    # plot examples from test set
    plot_couple_examples(
        model, test_loader, config.CONF_THRESHOLD,
        config.NMS_IOU_THRESH, config.ANCHORS, config.DEVICE
    )

if __name__ == '__main__':
    main()
