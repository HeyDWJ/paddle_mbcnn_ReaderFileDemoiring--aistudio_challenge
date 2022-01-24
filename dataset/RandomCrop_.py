import numpy as np
from PIL import Image
from paddle.vision.transforms import functional as F
import matplotlib.pyplot as plt

import glob, os

class RandomCrop():
    def __init__(self):
        super(RandomCrop).__init__()
    def get_params(self, img, output_size=(128, 128)):
        # 这里的image对象长什么样子
        # img_shape = img.size # w, h = img.size
        img_shape = img.shape
        left = np.random.randint(img_shape[2] - output_size[1])
        top = np.random.randint(img_shape[1] - output_size[0])
        bbox = top, left, output_size[0], output_size[1]
        return bbox
