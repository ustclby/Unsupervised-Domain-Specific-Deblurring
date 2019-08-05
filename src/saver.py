import os
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, Compose
# tensor to PIL Image
def tensor2img(img):
  img = img[0].cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
  return img.astype(np.uint8)

# save a set of images
def save_imgs(imgs, names, path, yuv=False):
    if not os.path.exists(path):
        os.mkdir(path)
    img = tensor2img(imgs)
    img = Image.fromarray(img)
    img.save(os.path.join(path, names))

class Saver():
  def __init__(self, opts):
    self.display_dir = os.path.join(opts.display_dir, opts.name)
    self.model_dir = os.path.join(opts.result_dir, opts.name)
    self.image_dir = os.path.join(self.model_dir, 'images')
    self.img_save_freq = opts.img_save_freq
    self.model_save_freq = opts.model_save_freq
    # make directory
    if not os.path.exists(self.display_dir):
      os.makedirs(self.display_dir)
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
      os.makedirs(self.image_dir)

  # save result images
  def write_img(self, ep, model):
    if (ep + 1) % self.img_save_freq == 0:
        assembled_images = model.assemble_outputs()
        img_filename = '%s/gen_%05d.jpg' % (self.image_dir, ep)
        torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
    elif ep == -1:
        assembled_images = model.assemble_outputs()
        img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
        torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

  # save model
  def write_model(self, ep, total_it, model):
    if (ep + 1) % self.model_save_freq == 0:
      print('--- save the model @ ep %d ---' % (ep))
      model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
    elif ep == -1:
      model.save('%s/last.pth' % self.model_dir, ep, total_it)

