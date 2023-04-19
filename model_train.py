import config
import torch
import copy 
from datetime import datetime

from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    check_class_accuracy,
    save_checkpoint
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, test_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    Train the model on the training set.
    
    Inputs:
    - train_loader: obj, torch.utils.data.DataLoader
    - test_loader: obj, torch.utils.data.DataLoader
    - model: obj, torch.nn.Module
    - optimizer: obj, torch.optim
    - loss_fn: obj, loss function
    - scaler: obj, torch.cuda.amp.GradScaler
    - scaled_anchors: tensor, shape (3, 3, 2)

    """
    map = 0
    best_model_wt = copy.deepcopy(model.state_dict())
    for epoch in range(config.NUM_EPOCHS):

        print('Epoch {}/{}'.format(epoch, config.NUM_EPOCHS - 1))
        
        loop = tqdm(train_loader, leave=True)
        losses = []

        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(config.DEVICE)
            y0, y1, y2 = (
                y[0].to(config.DEVICE),
                y[1].to(config.DEVICE),
                y[2].to(config.DEVICE),
            )   

            with torch.cuda.amp.autocast():
                # output shape (batch_size, 3, grid_size, grid_size, classes + 5)
                out = model(x)
                # calculate loss for each scale prediction and anchor box
                loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
                )

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)

        if epoch > 0 and epoch % 10 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
                device=config.DEVICE,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            cur_map = mapval.item()
            print(f"MAP: {cur_map}")
            if cur_map > map:
                best_model_wt = copy.deepcopy(model.state_dict())
                map = cur_map
                save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)
            model.train()

        torch.cuda.empty_cache()

    now = datetime.now()
    current_time = now.strftime("%H%M%S")
    model.load_state_dict(best_model_wt)
    torch.save(model, "tuned_model/yolov3-" + current_time + ".pth")
