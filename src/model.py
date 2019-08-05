import networks
import torch
import torch.nn as nn
import time


class UID(nn.Module):
  def __init__(self, opts):
    super(UID, self).__init__()

    # parameters
    lr = opts.lr
    self.nz = 8
    self.concat = opts.concat
    self.lambdaB = opts.lambdaB
    self.lambdaI = opts.lambdaI

    # discriminators
    if opts.dis_scale > 1:
      self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    else:
      self.disA = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
        
    # encoders
    self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b)
    if self.concat:
      self.enc_a = networks.E_attr_concat(opts.input_dim_b, self.nz, \
          norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu'))
    else:
      self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)

    # generator
    if self.concat:
      self.gen = networks.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
    else:
      self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

    # optimizers
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()
    if opts.percep == 'default':
        self.perceptualLoss = networks.PerceptualLoss(nn.MSELoss(), opts.gpu, opts.percp_layer)
    elif opts.percep == 'face':
        self.perceptualLoss = networks.PerceptualLoss16(nn.MSELoss(), opts.gpu, opts.percp_layer)
    else:
        self.perceptualLoss = networks.MultiPerceptualLoss(nn.MSELoss(), opts.gpu)

  def initialize(self):
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.disA2.apply(networks.gaussian_weights_init)
    self.disB2.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
    self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
    self.disA2_sch = networks.get_scheduler(self.disA2_opt, opts, last_ep)
    self.disB2_sch = networks.get_scheduler(self.disB2_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.disA.cuda(self.gpu)
    self.disB.cuda(self.gpu)
    self.disA2.cuda(self.gpu)
    self.disB2.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  def test_forward(self, image, a2b=True):
    if a2b:
        self.z_content = self.enc_c.forward_b(image)
        self.mu_b, self.logvar_b = self.enc_a.forward(image)
        std_b = self.logvar_b.mul(0.5).exp_()
        eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
        self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
        output = self.gen.forward_D(self.z_content, self.z_attr_b)
    
    return output


  def forward(self):
    # input images
    real_I = self.input_I
    real_B = self.input_B
    half_size = real_I.size(0) // 2
    self.real_I_encoded = real_I[0:half_size]
    self.real_I_random = real_I[half_size:]
    self.real_B_encoded = real_B[0:half_size]
    self.real_B_random = real_B[half_size:]

    # get encoded z_c
    self.z_content_i, self.z_content_b = self.enc_c.forward(self.real_I_encoded, self.real_B_encoded)

    # get encoded z_a
    if self.concat:
      self.mu_b, self.logvar_b = self.enc_a.forward(self.real_B_encoded)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_i, self.z_attr_b = self.enc_a.forward(self.real_I_encoded, self.real_B_encoded)

    # get random z_a
    self.z_random = self.get_z_random(self.real_I_encoded.size(0), self.nz, 'gauss')

    # first cross translation
    input_content_forI = torch.cat((self.z_content_b, self.z_content_i, self.z_content_b),0)
    input_content_forB = torch.cat((self.z_content_i, self.z_content_b, self.z_content_i),0)
    input_attr_forI = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)
    input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)


    output_fakeI = self.gen.forward_D(input_content_forI, input_attr_forI)
    output_fakeB = self.gen.forward_B(input_content_forB, input_attr_forB)
    self.fake_I_encoded, self.fake_II_encoded, self.fake_I_random = torch.split(output_fakeI, self.z_content_i.size(0), dim=0)
    self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_i.size(0), dim=0)

    # get reconstructed encoded z_c
    self.z_content_recon_b, self.z_content_recon_i = self.enc_c.forward(self.fake_I_encoded, self.fake_B_encoded)

    # get reconstructed encoded z_a
    if self.concat:
      self.mu_recon_b, self.logvar_recon_b = self.enc_a.forward(self.fake_B_encoded)
      std_b = self.logvar_recon_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_recon_b = eps_b.mul(std_b).add_(self.mu_recon_b)
    else:
      self.z_attr_recon_i, self.z_attr_recon_b = self.enc_a.forward(self.fake_A_encoded, self.fake_B_encoded)

    # second cross translation
    self.fake_I_recon = self.gen.forward_D(self.z_content_recon_i, self.z_attr_recon_b)
    self.fake_B_recon = self.gen.forward_B(self.z_content_recon_b, self.z_attr_recon_b)


    # for latent regression
    if self.concat:
      self.mu2_b, _ = self.enc_a.forward(self.fake_B_random)
    else:
      self.z_attr_random_i, self.z_attr_random_b = self.enc_a.forward(self.fake_B_random)


  def update_D(self, image_a, image_b):
    self.input_I = image_a
    self.input_B = image_b
    self.forward()

    # update disA
    self.disA_opt.zero_grad()
    loss_D1_A = self.backward_D(self.disA, self.real_I_encoded, self.fake_I_encoded)
    self.disA_loss = loss_D1_A.item()
    self.disA_opt.step()

    # update disA2
    self.disA2_opt.zero_grad()
    loss_D2_A = self.backward_D(self.disA2, self.real_I_random, self.fake_I_random)
    self.disA2_loss = loss_D2_A.item()
    self.disA2_opt.step()

    # update disB
    self.disB_opt.zero_grad()
    loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
    self.disB_loss = loss_D1_B.item()
    self.disB_opt.step()

    # update disB2
    self.disB2_opt.zero_grad()
    loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
    self.disB2_loss = loss_D2_B.item()
    self.disB2_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake = netD.forward(fake.detach())
    pred_real = netD.forward(real)
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D += ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def update_EG(self):
    # update G, Ec, Ea
    self.enc_c_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.enc_c_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()

    # update G, Ec
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_G_alone()
    self.enc_c_opt.step()
    self.gen_opt.step()

  def backward_EG(self):


    # Ladv for generator
    loss_G_GAN_I = self.backward_G_GAN(self.fake_I_encoded, self.disA)
    loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)

    # KL loss - z_a
    if self.concat:
      kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
      loss_kl_za_b = torch.sum(kl_element_b).mul_(-0.5) * 0.01
    else:
      loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01

    # cross cycle consistency loss
    loss_G_L1_I = self.criterionL1(self.fake_I_recon, self.real_I_encoded) * 10
    loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
    loss_G_L1_II = self.criterionL1(self.fake_II_encoded, self.real_I_encoded) * 10
    loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10
    
    # perceptual losses
    percp_loss_B = self.perceptualLoss.getloss(self.fake_I_encoded, self.real_B_encoded) * self.lambdaB
    percp_loss_I = self.perceptualLoss.getloss(self.fake_B_encoded, self.real_I_encoded) * self.lambdaI

    loss_G = loss_G_GAN_I + loss_G_GAN_B + \
             loss_G_L1_II + loss_G_L1_BB + \
             loss_G_L1_I + loss_G_L1_B + \
             loss_kl_za_b + percp_loss_B + \
             percp_loss_I

    loss_G.backward(retain_graph=True)

    self.gan_loss_i = loss_G_GAN_I.item()
    self.gan_loss_b = loss_G_GAN_B.item()

    self.kl_loss_za_b = loss_kl_za_b.item()
    self.l1_recon_I_loss = loss_G_L1_I.item()
    self.l1_recon_B_loss = loss_G_L1_B.item()
    self.l1_recon_II_loss = loss_G_L1_II.item()
    self.l1_recon_BB_loss = loss_G_L1_BB.item()
    self.B_percp_loss = percp_loss_B.item()
    self.G_loss = loss_G.item()


  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    loss_G = 0
    for out_a in outs_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    return loss_G

  def backward_G_alone(self):
    # Ladv for generator
    loss_G_GAN2_I = self.backward_G_GAN(self.fake_I_random, self.disA2)
    loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disB2)

    # latent regression loss
    if self.concat:
      loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
    else:
      loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10
    
    # perceptual losses
    percp_loss_B2 = self.perceptualLoss.getloss(self.fake_I_random, self.real_B_encoded) * self.lambdaB
    percp_loss_I2 = self.perceptualLoss.getloss(self.fake_B_random, self.real_I_encoded) * self.lambdaI


    loss_G2 = loss_z_L1_b + loss_G_GAN2_I + loss_G_GAN2_B + percp_loss_B2 + percp_loss_I2
    loss_G2.backward()
    self.gan2_loss_a = loss_G_GAN2_I.item()
    self.gan2_loss_b = loss_G_GAN2_B.item()

  def update_lr(self):
    self.disA_sch.step()
    self.disB_sch.step()
    self.disA2_sch.step()
    self.disB2_sch.step()
    self.enc_c_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)

    # weight
    if train:
      self.disA.load_state_dict(checkpoint['disA'])
      self.disA2.load_state_dict(checkpoint['disA2'])
      self.disB.load_state_dict(checkpoint['disB'])
      self.disB2.load_state_dict(checkpoint['disB2'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    if train:
      self.disA_opt.load_state_dict(checkpoint['disA_opt'])
      self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
      self.disB_opt.load_state_dict(checkpoint['disB_opt'])
      self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
             'disA': self.disA.state_dict(),
             'disA2': self.disA2.state_dict(),
             'disB': self.disB.state_dict(),
             'disB2': self.disB2.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'disA_opt': self.disA_opt.state_dict(),
             'disA2_opt': self.disA2_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'disB2_opt': self.disB2_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'ep': ep,
             'total_it': total_it
              }
    time.sleep(10)
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_I_encoded).detach()
    images_b = self.normalize_image(self.real_B_encoded).detach()
    images_a1 = self.normalize_image(self.fake_I_encoded).detach()
    images_a2 = self.normalize_image(self.fake_I_random).detach()
    images_a3 = self.normalize_image(self.fake_I_recon).detach()
    images_a4 = self.normalize_image(self.fake_II_encoded).detach()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach()
    images_b2 = self.normalize_image(self.fake_B_random).detach()
    images_b3 = self.normalize_image(self.fake_B_recon).detach()
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
    row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
