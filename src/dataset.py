import copy
import json
import os
import random
import cv2
import numpy as np
import scipy
import torch
import torchvision.transforms.functional as F
from PIL import Image
# from scipy.misc import imread
from skimage.color import rgb2gray, gray2rgb

from torch.utils.data import DataLoader

class DataAugment:
    def __init__(self,debug=False):
        self.debug=debug
        # print("Data augment...")

    def basic_matrix(self,translation):
        """基础变换矩阵"""
        return np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

    def adjust_transform_for_image(self,img,trans_matrix):
        """根据图像调整当前变换矩阵"""
        transform_matrix=copy.deepcopy(trans_matrix)
        height, width, channels = img.shape
        transform_matrix[0:2, 2] *= [width, height]
        center = np.array((0.5 * width, 0.5 * height))
        transform_matrix = np.linalg.multi_dot([self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        return transform_matrix

    def apply_transform(self,img,transform):
        """仿射变换"""
        output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=0,)   #cv2.BORDER_REPLICATE,cv2.BORDER_TRANSPARENT
        return output

    def apply(self,img,trans_matrix):
        """应用变换"""
        tmp_matrix=self.adjust_transform_for_image(img, trans_matrix)
        out_img=self.apply_transform(img, tmp_matrix)
        if self.debug:
            self.show(out_img)
        return out_img

    def random_vector(self,min,max):
        """生成范围矩阵"""
        min=np.array(min)
        max=np.array(max)
        # print(min.shape,max.shape)
        assert min.shape==max.shape
        assert len(min.shape) == 1
        return np.random.uniform(min, max)

    def show(self,img):
        """可视化"""
        cv2.imshow("outimg",img)
        cv2.waitKey()

    def random_transform(self,img,min_translation,max_translation):
        """平移变换"""
        factor=self.random_vector(min_translation,max_translation)
        trans_matrix=np.array([[1, 0, factor[0]],[0, 1, factor[1]],[0, 0, 1]])
        return trans_matrix


    def random_rotate(self,img,factor):
        """随机旋转"""
        angle=np.random.uniform(factor[0],factor[1])
        # print("angle:{}".format(angle))
        rotate_matrix=np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])

        return rotate_matrix

    def random_scale(self,img,min_translation,max_translation):
        """随机缩放"""
        factor=self.random_vector(min_translation, max_translation)
        scale_matrix = np.array([[factor[0], 0, 0],[0, factor[1], 0],[0, 0, 1]])

        return scale_matrix

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.mask = config.MASK
        self.nms = config.NMSMASK_REVERSE

        self.reverse_mask = config.MASK_REVERSE
        self.mask_threshold = config.MASK_THRESHOLD

        print('training:{}  mask:{}  mask_list:{}  data_list:{}'.format(training, self.mask, mask_flist, flist))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item= self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item= self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        # img = imread(self.data[index])
        img = np.load(self.data[index])
        #南极归1化
        scale_rate = 1.5
        imgmin = img[img != 0].min()
        imgmax = img[img != 0].max()
        imglow = ((imgmax + imgmin) - (imgmax - imgmin) * scale_rate) / 2
        imghigh = ((imgmax + imgmin) + (imgmax - imgmin) * scale_rate) / 2
        imgrange = (imghigh - imglow)
        if imgrange != 0:
            img = ((img - imglow) / imgrange)
        else:
            imgrange = 1
            img = ((img - imglow) / imgrange)

        #逆归一化
        #SR = sr*imgrange + imglow

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # demo = DataAugment(debug=False)
        # t1 = demo.random_transform(img, (-0.3, -0.3), (0.3, 0.3))
        # t2 = demo.random_rotate(img, (0.5, 0.8))
        # t3 = demo.random_scale(img, (1.5, 1.5), (1.7, 1.7))
        # tmp = np.linalg.multi_dot([t1, t2, t3])
        # img = demo.apply(img, tmp)

        # load mask
        mask = self.load_mask(img, index % len(self.mask_data))

        if self.reverse_mask == 1:
            mask = 255 - mask


        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]


        return self.to_tensor(img), self.to_tensor(mask), imgrange, imglow


    def load_mask(self, img, index): 
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        if self.training:
            mask_index = random.randint(0, len(self.mask_data) - 1)
        else:
            mask_index = index
            print('+++++++++++++++')

        # mask = imread(self.mask_data[mask_index])
        mask = np.load(self.mask_data[mask_index])
        mask = self.resize(mask, imgh, imgw)
        mask = (mask > self.mask_threshold).astype(np.uint8) * 255       # threshold due to interpolation


        return mask

    def to_tensor(self, img):
        # img = Image.fromarray(img)
        img = np.ascontiguousarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        # img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if flist is None:
            return []
        with open(flist, 'r') as j:
            f_list = json.load(j)
            return f_list


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
