"""
Output frames (with specified framerate)
"""
import os
import numpy as np
from matplotlib.pyplot import *
import cv2

rate = 10
DATASET = os.path.expanduser("~/data/50Salads/")
url_videos = DATASET+"raw/rgb/"
url_imgs = DATASET+"raw/imgs/"

files = os.listdir(url_videos)
files = [f for f in files if f[0]!="."]
files = np.sort(files)

for f in files:
    vid = cv2.VideoCapture(url_videos+f)
    ret, im = vid.read()
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    if not os.path.exists(url_imgs+f):
        os.mkdir(url_imgs+f)

    for i in xrange(n_frames/rate):
        print(i, "of", n_frames/rate)
        if i > n_frames:
            print("New video")
            break

        vid.set(cv2.CAP_PROP_POS_FRAMES, i*rate)
        ret, im = vid.read()
        if not ret:
            print("No image")
            break

        # im = cv2.resize(im, (256,256))

        cv2.imshow("im", im)
        ret = cv2.waitKey(1)
        if ret >= 0:
            break

        # cv2.imwrite("{}{}/{}_{}.jpg".format(url_imgs, f, f, i*rate), im)
