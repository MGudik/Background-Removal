
from mrcnn import model as modellib
import config as CONFIG
import cv2
import numpy as np


class Detector():

    model = modellib.MaskRCNN(
        mode="inference", config=CONFIG.DETECTION_CONFIG, model_dir='models/')
    model.load_weights(CONFIG.MODEL, by_name=True)

    def __call__(self, image):
        if type(image) != list:
            image = [image]
        predictions = self.model.detect(image)[0]

        masks = predictions['masks']
        classes = predictions['class_ids']
        boxes = predictions['rois']

        allowed_indices = [c in CONFIG.ALLOWED_CLASSES for c in classes]

        masks = masks[:, :, allowed_indices]
        classes = classes[allowed_indices]
        boxes = boxes[allowed_indices, :]

        return masks, classes

    def masks_to_img(self, masks):
        image = np.zeros((masks.shape[0], masks.shape[1]), dtype=np.uint8)
        for idx in range(masks.shape[-1]):
            mask = masks[:, :, idx]
            contours, hierarchy = cv2.findContours(mask.astype(
                np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, (255, 255, 255), -1)

        return image

    def blur_unmasked(self, image, masks):
        raw_img = image.copy()
        image = cv2.GaussianBlur(image, (15, 15), 0)
        for idx in range(masks.shape[-1]):
            mask = masks[:, :, idx]
            image[mask == 1] = raw_img[mask == 1]
        return image

    def black_unmasked(self, image, masks):

        raw_image = image.copy()
        image = np.zeros_like(raw_image)
        for idx in range(masks.shape[-1]):
            mask = masks[:, :, idx]
            image[mask == 1] = raw_image[mask == 1]
        return image
