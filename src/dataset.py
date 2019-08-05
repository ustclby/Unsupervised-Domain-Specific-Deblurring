import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random

class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(self.dataroot)
    self.img = [os.path.join(self.dataroot, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim
    transforms = [ToTensor()]
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img, img_name

  def __len__(self):
    return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot
    # A
    images_A = sorted(os.listdir(os.path.join(self.dataroot, opts.phase + 'A')))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]
    # B
    images_B = sorted(os.listdir(os.path.join(self.dataroot, opts.phase + 'B')))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b
    self.resize_x = opts.resize_size_x
    self.resize_y = opts.resize_size_y

    if opts.phase == 'train':
      transforms = [RandomCrop(opts.crop_size)]
    else:
      transforms = [CenterCrop(opts.crop_size)]
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):   
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      temp_b_index = random.randint(0, self.B_size - 1)
      data_B = self.load_img(self.B[temp_b_index], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
        
    img = Image.open(img_name).convert('RGB')
    (w,h) = img.size
    if w < h:
        resize_x = self.resize_x
        resize_y = round(self.resize_x * h / w)
    else:
        resize_y = self.resize_y
        resize_x = round(self.resize_y * w / h)
    resize_img = Compose([Resize((resize_y, resize_x), Image.BICUBIC)])
    img = resize_img(img)
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size
