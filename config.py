import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from utils import seed_everything

DATASET = 'defect_image'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_RATIO = 0.8
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 480
NUM_CLASSES = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.5 # > confidence threshold, we consider object as detected
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block (conv + conv + residual) followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
model_architecture = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S", # Scale prediction block 13*13
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S", # Scale prediction block 26*26
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S", # Scale prediction block 52*52
]

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]
# anchors[0] for first scale


scale = 1.1
train_transforms = A.Compose(
    [
        # A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        # A.PadIfNeeded(
        #     min_height=int(IMAGE_SIZE * scale),
        #     min_width=int(IMAGE_SIZE * scale),
        #     border_mode=cv2.BORDER_CONSTANT,
        # ),
        # A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        # A.OneOf(
        #     [
        #         A.ShiftScaleRotate(
        #             rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
        #         ),
        #         A.IAAAffine(shear=15, p=0.5, mode="constant"),
        #     ],
        #     p=1.0,
        # ),
        # A.HorizontalFlip(p=0.5),
        # A.Blur(p=0.1),
        # A.CLAHE(p=0.1),
        # A.Posterize(p=0.1),
        # A.ToGray(p=0.1),
        # A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        # A.LongestMaxSize(max_size=IMAGE_SIZE),
        # A.PadIfNeeded(
        #     min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        # ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

CLASSES = [
    "defect",
    "non-defect"
]
