import os
from mrcnn.config import Config


####################################################
###               Model parameters               ###
####################################################
MODEL = r'models/mask_rcnn_coco.h5'
DETECTION_THRESHOLD = 0.8
ALLOWED_CLASSES = [1]


####################################################
###           Visualization parameters           ###
####################################################


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class InferenceConfig(Config):
    GPU_COUNT = 1
    NAME = 'coco'
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = DETECTION_THRESHOLD
    NUM_CLASSES = 1 + 80
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


DETECTION_CONFIG = InferenceConfig()
