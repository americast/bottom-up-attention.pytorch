import numpy as np
import os
import random
from copy import deepcopy
import cv2
from tqdm import tqdm
import lmdb
import pudb
import pickle
import base64

random.seed(0)
files_match = []
files = []
os.system("rm vis_res_pytorch/*")


env = lmdb.open("/srv/share3/amoudgil3/Recurrent-VLN-BERT/img_features/matterport-ResNet-101-faster-rcnn-genome.lmdb", readonly=True)

txn = env.begin()
boxes_caffe = []
boxes_view = []
i = 0
print("\nExtracting...")
for key, value in tqdm(txn.cursor()):
    files_match.append(str(key)[2:-1])
    val_dict = pickle.loads(value)

    j = val_dict["boxes"]
    l = np.frombuffer(base64.b64decode(j), dtype=np.float32).reshape((-1, 4))
    boxes_caffe.append(l)

    j = val_dict["featureViewIndex"]
    l = np.frombuffer(base64.b64decode(j), dtype=np.float32)
    boxes_view.append(l)

    i += 1
    if i == 100:
        break

imgs = {}
for i, f in enumerate(files_match):
    f = f.replace("-","_")
    fs = [f+"_"+str(int(x)) for x in boxes_view[i]]
    files.extend(fs)

print("\nGenerating...")
for i, boxes in enumerate(tqdm(boxes_caffe)):
    file_here = files_match[i].replace("-","_")
    for idx in boxes_view[i]:
        file_here_idx = file_here+"_"+str(int(idx))
        if file_here_idx not in imgs:
            img_0 = cv2.imread("../Matterport3DSimulator/scene_imgs_now_again_0/"+file_here_idx+".png")
            img_1 = cv2.imread("../Matterport3DSimulator/scene_imgs_now_again_1/"+file_here_idx+".png")
            img_2 = cv2.imread("../Matterport3DSimulator/scene_imgs_now_again_2/"+file_here_idx+".png")
            img_3 = cv2.imread("../Matterport3DSimulator/scene_imgs_now_again_3/"+file_here_idx+".png")
            
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

            imgs[file_here_idx] = img
            
        bbox = list(boxes[i].astype("int"))
        imgs[file_here_idx] = cv2.rectangle(imgs[file_here_idx], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0))

print("\nWriting...")
for each in tqdm(imgs):
    img = imgs[each]
    cv2.imwrite("vis_res_caffe/"+each+".png", img)
    
# # img_name = f.split(".")[0]+".png"

# img_npz = deepcopy(img)
# meta_here = np.load("scene_imgs_now_feats/"+f, allow_pickle=True)
# bboxes = meta_here["bbox"]
# for bbox in bboxes:
#     img_npz = cv2.rectangle(img_npz, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0))


