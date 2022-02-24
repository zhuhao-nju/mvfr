import cv2
import os
import tqdm

# in_dir = '/home/yht/share/xyz_new/toYHT/filter/'
# out_dir = '/home/yht/share/xyz_new/toYHT/post_process/filter2/'
#
# for dir in tqdm.tqdm(os.listdir(in_dir)):
#     dpmap = cv2.imread(in_dir + dir + '/cha_REC_f.tif')
#     for name in os.listdir(in_dir + dir):
#         if name[-3:] == 'png':
#             break
#     cv2.imwrite(out_dir + name[8:], dpmap[::-1])

# in_dir = '/home/yht/share/xyz_new/toYHT/MVS_face_dpmap_1024_relocated/'
# out_dir = '/home/yht/share/xyz_new/toYHT/post_process/pred/'
in_dir = '/home/yht/share/xyz_new/toYHT/paper_demand_dpmap/'
out_dir = '/home/yht/share/xyz_new/toYHT/paper_demand_dpmap/'

from PIL import Image
import numpy as np

for name in tqdm.tqdm(os.listdir(in_dir)):
    dpmap = Image.open(in_dir + name)
    dpmap = np.array(dpmap)[1600 - 1024:1600 + 1024, 2048 - 1024:2048 + 1024]
    dpmap[:] = dpmap[::-1]
    dpmap = Image.fromarray(dpmap)
    dpmap.save(out_dir + name)
