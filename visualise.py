import numpy as np
import os
import random
from copy import deepcopy
import cv2
from tqdm import tqdm

files = os.listdir("scene_imgs_now_feats")
random.shuffle(files)
files = files[:100]

for f in tqdm(files):
    img_name = f.split(".")[0]+".png"
    
    img_0 = cv2.imread("../Matterport3DSimulator/scene_imgs_now_again_0/"+img_name)
    img_1 = cv2.imread("../Matterport3DSimulator/scene_imgs_now_again_1/"+img_name)
    img_2 = cv2.imread("../Matterport3DSimulator/scene_imgs_now_again_2/"+img_name)
    img_3 = cv2.imread("../Matterport3DSimulator/scene_imgs_now_again_3/"+img_name)
    
    try:
        img_0.shape
        img = img_0
    except:
        pass

    try:
        img_1.shape
        img = img_1
    except:
        pass
    
    try:
        img_2.shape
        img = img_2
    except:
        pass

    try:
        img_3.shape
        img = img_3
    except:
        pass
    
    img_npz = deepcopy(img)
    meta_here = np.load("scene_imgs_now_feats/"+f, allow_pickle=True)
    bboxes = meta_here["bbox"]
    for bbox in bboxes:
        img_npz = cv2.rectangle(img_npz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0))
    cv2.imwrite("vis_res_pytorch/"+img_name, img_npz)


