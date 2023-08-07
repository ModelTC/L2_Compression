
import os
import torch
import torchvision.transforms as T
# import cv2
from torch.utils.data import Dataset
import torch
from PIL import Image


class ImageNetDataset(Dataset):
    """
    ImageNet Dataset.
    """
    def __init__(self, root_dir, meta_file, transform):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
    
        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            filename, label = line.rstrip().split()
            self.metas.append({'filename': filename, 'label': label})


    def __len__(self):
        return self.num

    def _load_meta(self, idx):

        return self.metas[idx]

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = os.path.join(self.root_dir, curr_meta['filename'])
        label = int(curr_meta['label'])
        with Image.open(filename) as img:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
        # item = {
        #     'image': img,
        #     'label': label,
        #     'image_id': idx,
        #     'filename': filename
        # }
        # return item
