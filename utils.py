import json
import os
import cv2


def parse_annotation(path):
    with open(path, 'r') as f:
        annotation = json.loads(f.read())
        return annotation


def make_dir_if_needed(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def resize_image(image, size=600):
    width = int(size * image.shape[1] / image.shape[0] if image.shape[0] > image.shape[1] else size)
    height = int(size * image.shape[0] / image.shape[1] if image.shape[0] < image.shape[1] else size)
    return cv2.resize(image, (width, height))
