import cv2
from utils import color_transfer, tensor2im, subdiv, dpmap2verts
import numpy as np
import os
from options import Options
from pix2pixHD.models import create_model
import torch
import torchvision.transforms.functional as F
from PIL import Image
import time


def main():
    opt = Options().parse()
    img_names = []
    for name in os.listdir(opt.input):
        if any(name.endswith(extension) for extension in
               ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff']):
            img_names.append(name)

    dpmap_model = create_model(opt)

    for img_name in img_names:
        print(f'\nProcessing {img_name}')
        base_name = os.path.splitext(img_name)[0]

        img = cv2.imread(f'{opt.input}/{img_name}')

        # start = time.time()

        texture = cv2.resize(img, (1024, 1024)).astype(np.uint8)

        # mask = (255 - cv2.imread(f'{opt.predef_dir}/front_mask.png')[:, :, 0]).astype(bool)
        # new_pixels = color_transfer(texture[mask][:, np.newaxis, :])
        # texture[mask] = new_pixels[:, 0, :]
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB).astype(np.float32)
        texture = np.transpose(texture, (2, 0, 1))
        texture = torch.tensor(texture) / 255
        texture = F.normalize(texture, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True)
        texture = torch.unsqueeze(texture, 0)

        print('Generating displacement maps...')
        dpmap_full = np.zeros((4096, 4096), dtype=np.uint16)
        dpmap_full[...] = 32768
        dpmap_full = Image.fromarray(dpmap_full)

        dpmap = dpmap_model.inference(texture, torch.tensor(0))
        dpmap = tensor2im(dpmap.detach()[0], size=(1900, 1900))
        dpmap = Image.fromarray(dpmap)
        dpmap_full.paste(dpmap, (1100, 600, 3000, 2500))
        # print(time.time() - start)
        dpmap_full.save(f'{opt.output}/{base_name}.png')


if __name__ == '__main__':
    main()
