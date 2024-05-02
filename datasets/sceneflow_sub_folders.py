# This file has been modified to adapt to Sceneflow stereo imgs dataset.
import os
import random
import numpy as np
import torch.utils.data as data

from PIL import Image
from . import readpfm as rp


def default_loader(path):
    return Image.open(path).convert('RGB')

def mask_loader(path):
    return np.array(Image.open(path))

def disparity_loader(path):
    return rp.readPFM(path)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    

class SceneflowSubDataset(data.Dataset):
    def __init__(self, root, transform=None, train=True):

        self.root = root
        self.train = train
        self.loader = default_loader
        self.maskloader = mask_loader
        self.dploader = disparity_loader
        self.transform = transform
        self.crawl_folders()

    def crawl_folders(self):
        self.left=[]
        self.right=[]
        self.disp_L = []
        self.disp_R = []
        self.mask_L = []
        self.mask_R = []

        # flying3d subset
        flying_path = self.root + 'frames_cleanpass'
        flying_disp = self.root + 'frames_disparity'
        flying_mask = self.root + 'disparity_occlusions'

        if self.train:
            
            flying_dir = flying_path+'/train/'

            imm_l = os.listdir(flying_dir + '/left')

            for im in imm_l:
                if is_image_file(flying_dir + '/left/' + im):
                    self.left.append(flying_dir + '/left/' + im)

                if is_image_file(flying_dir + '/right/' + im):
                    self.right.append(flying_dir + '/right/' + im)
                
                self.disp_L.append(flying_disp+'/train/'+'/left/'+im.split(".")[0]+'.pfm')
                self.disp_R.append(flying_disp+'/train/'+'/right/'+im.split(".")[0]+'.pfm')
                self.mask_L.append(flying_mask+'/train/'+'/left/'+im)
                self.mask_R.append(flying_mask+'/train/'+'/right/'+im)

        else:

            flying_dir = flying_path+'/val/'

            imm_l = os.listdir(flying_dir + '/left')

            for im in imm_l:
                if is_image_file(flying_dir + '/left/' + im):
                    self.left.append(flying_dir + '/left/' + im)

                if is_image_file(flying_dir + '/right/' + im):
                    self.right.append(flying_dir + '/right/' + im)
                
                self.disp_L.append(flying_disp+'/val/'+'/left/'+im.split(".")[0]+'.pfm')
                self.disp_R.append(flying_disp+'/val/'+'/right/'+im.split(".")[0]+'.pfm')
                self.mask_L.append(flying_mask+'/val/'+'/left/'+im)
                self.mask_R.append(flying_mask+'/val/'+'/right/'+im)


    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]
        disp_R= self.disp_R[index]
        mask_L= self.mask_L[index]
        mask_R= self.mask_R[index]


        left_img = self.loader(left)
        right_img = self.loader(right)
        left_mask = self.maskloader(mask_L)
        right_mask = self.maskloader(mask_R)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)
        dataR, scaleR = self.dploader(disp_R)
        dataR = np.ascontiguousarray(dataR,dtype=np.float32)

        # if self.train:  
        #     w, h = left_img.size
        #     th, tw = 256, 512

        #     x1 = random.randint(0, w - tw)
        #     y1 = random.randint(0, h - th)

        #     left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
        #     right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

        #     dataL = dataL[y1:y1 + th, x1:x1 + tw]
        #     left_mask = left_mask[y1:y1 + th, x1:x1 + tw]
        #     right_mask = right_mask[y1:y1 + th, x1:x1 + tw]
 
        #     left_img   = self.transform(left_img)
        #     right_img  = self.transform(right_img)

        # else: 
        left_img = left_img.crop((0, 0, 576, 512))
        right_img = right_img.crop((0, 0, 576, 512))

        dataL = dataL[0:512, 0:576]
        dataR = dataR[0:512, 0:576]
        left_mask = left_mask[0:512, 0:576]
        right_mask = right_mask[0:512, 0:576]

        left_img       = self.transform(left_img)
        right_img      = self.transform(right_img) 

        return left_img, right_img, dataL, dataR, left_mask, right_mask

    def __len__(self):
        return len(self.left)
