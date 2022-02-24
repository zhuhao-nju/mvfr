import os
import shutil

in_dir = '/home/yht/share/xyz_new/toYHT/MVS_face_texture_1024_relocated/'
out_dir = '/home/yht/share/xyz_new/toYHT/filter/'

for i, img in enumerate(os.listdir(in_dir)):
    os.mkdir(out_dir + str(i + 1))
    shutil.copyfile(in_dir + img, out_dir + str(i + 1) + '/' + img)
