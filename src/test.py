import os
import numpy as np
import torch
import torch.nn as nn

from dataset import dataset_single
from model import UID
from networks import PerceptualLoss16,PerceptualLoss
from options import TestOptions
from saver import save_imgs
from shutil import copyfile
from skimage.measure import compare_psnr as PSNR
from skimage.measure import compare_ssim as SSIM
from skimage.io import imread
from skimage.transform import resize

def main():
    
    # parse options
    parser = TestOptions()
    opts = parser.parse()
    result_dir = os.path.join(opts.result_dir, opts.name)
    orig_dir = opts.orig_dir
    blur_dir = opts.dataroot

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # data loader
    print('\n--- load dataset ---')
    if opts.a2b:
        dataset = dataset_single(opts, 'A', opts.input_dim_a)
    else:
        dataset = dataset_single(opts, 'B', opts.input_dim_b)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = UID(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    # test
    print('\n--- testing ---')
    for idx1, (img1,img_name) in enumerate(loader):
        print('{}/{}'.format(idx1, len(loader)))
        img1 = img1.cuda(opts.gpu).detach()
        with torch.no_grad():
            img = model.test_forward(img1, a2b=opts.a2b)
        img_name = img_name[0].split('/')
        img_name = img_name[-1]
        save_imgs(img, img_name, result_dir)
  
     # evaluate metrics
    if opts.percep == 'default':
        pLoss = PerceptualLoss(nn.MSELoss(),p_layer=36)
    elif opts.percep == 'face':
        self.perceptualLoss = networks.PerceptualLoss16(nn.MSELoss(),p_layer=30)
    else:
        self.perceptualLoss = networks.MultiPerceptualLoss(nn.MSELoss())
    
    orig_list = sorted(os.listdir(orig_dir))
    deblur_list = sorted(os.listdir(result_dir)) 
    blur_list = sorted(os.listdir(blur_dir)) 
    
    psnr = []
    ssim = []
    percp = []
    blur_psnr = []
    blur_ssim = []
    blur_percp = []

    for (deblur_img_name, orig_img_name, blur_img_name) in zip(deblur_list, orig_list, blur_list):
        deblur_img_name = os.path.join(result_dir,deblur_img_name)
        orig_img_name = os.path.join(orig_dir,orig_img_name)
        blur_img_name = os.path.join(blur_dir, blur_img_name)
        deblur_img = imread(deblur_img_name)
        orig_img = imread(orig_img_name)
        blur_img = imread(blur_img_name)
        try:
            psnr.append(PSNR(deblur_img, orig_img))
            ssim.append(SSIM(deblur_img, orig_img, multichannel=True))
            blur_psnr.append(PSNR(blur_img, orig_img))
            blur_ssim.append(SSIM(blur_img, orig_img, multichannel=True))
        except ValueError:
            print(orig_img_name)
        
        with torch.no_grad():
            temp = pLoss.getloss(deblur_img,orig_img)
            temp2 = pLoss.getloss(blur_img,orig_img)
        percp.append(temp)
        blur_percp.append(temp2)
        
    print(sum(psnr)/len(psnr))
    print(sum(ssim)/len(ssim))
    print(sum(percp)/len(percp))
    
    print(sum(blur_psnr)/len(psnr))
    print(sum(blur_ssim)/len(ssim))
    print(sum(blur_percp)/len(percp))
    return

if __name__ == '__main__':
  main()
