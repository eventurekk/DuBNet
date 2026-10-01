import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from pdb import set_trace as stx
import random
import cv2
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):  # # rgb_dir  = ./Datasets/train/, img_options = {'patch_size':256}
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))  # inp_files = ./Datasets/train/input
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))  # tar_files = ./Datasets/train/target

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        # inp_img = cv2.imread(inp_path).astype(np.float32)
        # tar_img = cv2.imread(tar_path).astype(np.float32)
        inp_img = cv2.imread(inp_path)
        tar_img = cv2.imread(tar_path)

        # x, y = inp_img.shape[1], inp_img.shape[2]
        # inp_img = cv2.resize(inp_img, None, fx=0.125, fy=0.125)
        # tar_img = cv2.resize(tar_img, None, fx=0.125, fy=0.125)

        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2YCrCb)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2YCrCb)

        # inp_img = Image.open(inp_path).convert('RGB')
        # tar_img = Image.open(tar_path).convert('RGB')
        # inp_img = load_img(inp_path)
        # tar_img = load_img(tar_path)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        # Crop Input and Target
        H, W = inp_img.shape[1], inp_img.shape[2]

        # if H - ps == 0:
        #     r = 0
        #     c = 0
        # else:
        #     r = np.random.randint(0, H - ps)
        #     c = np.random.randint(0, W - ps)
        r = 0 if H == ps else np.random.randint(0, H - ps + 1)
        c = 0 if W == ps else np.random.randint(0, W - ps + 1)
        inp_img = inp_img[:, r:r + ps, c:c + ps]
        tar_img = tar_img[:, r:r + ps, c:c + ps]


        # Data Augmentations
        aug = random.randint(0, 8)
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        # DataLoaderTrain/DataLoaderVal 的 __getitem__ 函数最后，return之前添加：
        # print(
        #     f"DataLoader输出 - 输入图(inp_img)：min={inp_img.min().item():.2f}, max={inp_img.max().item():.2f}, 均值={inp_img.mean().item():.2f}")
        # print(
        #     f"DataLoader输出 - 目标图(tar_img)：min={tar_img.min().item():.2f}, max={tar_img.max().item():.2f}, 均值={tar_img.mean().item():.2f}")
        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        # inp_img = cv2.imread(inp_path).astype(np.float32)
        # tar_img = cv2.imread(tar_path).astype(np.float32)
        inp_img = cv2.imread(inp_path)
        tar_img = cv2.imread(tar_path)

        height, width = inp_img.shape[0], inp_img.shape[1]
        height = round(height / 8) * 8
        width = round(width / 8) * 8

        inp_img = cv2.resize(inp_img, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
        tar_img = cv2.resize(tar_img, dsize=(width, height), interpolation=cv2.INTER_NEAREST)

        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2YCrCb)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2YCrCb)

        # 先转成 tensor
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        # 再 center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, [ps, ps])
            tar_img = TF.center_crop(tar_img, [ps, ps])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index_):
        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        # inp_img = Image.open(inp_path).convert('RGB')
        # tar_img = Image.open(tar_path).convert('RGB')

        # inp_img = cv2.imread(inp_path).astype(np.float32)
        # tar_img = cv2.imread(tar_path).astype(np.float32)
        inp_img = cv2.imread(inp_path)
        tar_img = cv2.imread(tar_path)

        height, width = inp_img.shape[0], inp_img.shape[1]
        height = round(height / 8) * 8
        width = round(width / 8) * 8
        inp_img = cv2.resize(inp_img, dsize=(width, height), dst=None, interpolation=cv2.INTER_NEAREST)
        tar_img = cv2.resize(tar_img, dsize=(width, height), dst=None, interpolation=cv2.INTER_NEAREST)

        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2YCrCb)
        tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2YCrCb)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderLow(Dataset):
    def __init__(self, rgb_dir, img_options):
        super(DataLoaderLow, self).__init__()

        inp_files = sorted(os.listdir(rgb_dir))

        self.inp_filenames = [os.path.join(rgb_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index_):
        inp_path = self.inp_filenames[index_]

        # inp_img = Image.open(inp_path).convert('RGB')

        # inp_img = cv2.imread(inp_path).astype(np.float32
        inp_img = cv2.imread(inp_path)
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2YCrCb)

        inp_img = TF.to_tensor(inp_img)


        # pad input image to be a multiple of window_size
        h_old, w_old = inp_img.shape[1], inp_img.shape[2]
        h_new = h_old - (h_old % 8)
        w_new = w_old - (w_old % 8)

        inp_img = TF.center_crop(inp_img, [h_new, w_new])


        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]

        return inp_img, filename

