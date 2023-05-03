import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CardDataset(Dataset):
    # CHARS = '0123456789'
    CHARS = '0123456789/'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, image_dir, mode, img_height, img_width):
        texts = []
        self.mode = mode
        self.image_dir = image_dir
        self.img_height = img_height
        self.img_width = img_width
        if mode == "train" or mode == "val":
            file_names, texts = self._load_from_raw_files()
            self.file_names = file_names
        self.texts = texts

    def _load_from_raw_files(self):
        file_names = []
        texts = []
        file_names = os.listdir(self.image_dir)
        for file_name in file_names:
            text = file_name.split('.')[0].split('-')[0].split(' ')[0]
            # texts.append(text.replace('_', ''))
            texts.append(text.replace('_', '/'))

        return file_names, texts

    def __len__(self):
        if self.mode == "pred":
            return 1
        else:
            return len(self.file_names)
        
    def __getitem__(self, index):
        if self.mode == "train" or self.mode == "val":
            file_name = self.file_names[index]
            file_path = os.path.join(self.image_dir,file_name)
            image = Image.open(file_path)
        elif self.mode == "pred":
            # 此时image_dir为PIL.Image对象
            image = self.image_dir

        # 图像预处理，转为PyTorch张量
        image = image.convert('L').resize((self.img_width, self.img_height))
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        # 归一化到[-1, 1]
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        if len(self.texts) != 0:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]
            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image

# 不定长数据集，重写collate_fn，图像使用stack而数据使用cat
def cardnumber_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
