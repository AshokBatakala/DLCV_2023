
#imports here
from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# default values for hyperparameters -------------------
ngpu = 1
nz = 100
ngf = 64
ndf = 64
nc = 3



# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.



#weights initialization--------------------------------

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# generator--------------------------------------------

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu = ngpu, nz = nz, ngf = ngf, nc = nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# discriminator----------------------------------------

class Discriminator(nn.Module):
    def __init__(self, ngpu = ngpu, nz = nz, ndf = ndf, nc = nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



# training function------------------------------------


def train(netG, netD, dataloader, criterion, optimizerD, optimizerG, device, num_epochs=5,fixed_noise=None, callbacks = None):
    """ Training function for DCGAN
    Args:
        netG (nn.Module): Generator network
        netD (nn.Module): Discriminator network
        dataloader (torch.utils.data.DataLoader): Dataloader for training data
        criterion (torch.nn.modules.loss): Loss function
        optimizerD (torch.optim): Optimizer for discriminator
        optimizerG (torch.optim): Optimizer for generator
        device (torch.device): Device to use for training
        num_epochs (int): Number of epochs to train for

    Returns:
        G_losses (list): List of generator losses
        D_losses (list): List of discriminator losses
        D_accuracy (list): List of discriminator accuracy
        img_list (list): List of fake images generated during training
    """

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_accuracy = []

    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):

        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # D accuracy
            D_accuracy.append((D_x - D_G_z2 +1) / 2) # ================================================================


            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if fixed_noise is not None:
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
        # Callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(epoch, netG, netD)

    return G_losses, D_losses, D_accuracy, img_list


    # ---------------------------------------------
    # generator wih AdaIN as in stylegan2
    # ---------------------------------------------

class mlp_2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(mlp_2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        return x

class AdaIN_Generator(nn.Module):
    def __init__(self, ngpu = ngpu, nz = nz, ngf = ngf, nc = nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

        self.z_to_w = nn.Sequential(
            nn.Linear(nz, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, nz)
        )

        self.fory1 = mlp_2(nz, 512*4*4, 512)

        self.forb1 = mlp_2(nz, 512*4*4, 512)


        self.fory2 = mlp_2(nz, 512*8*8, 512)

        self.forb2 = mlp_2(nz, 512*8*8, 512)

        self.fory3 = mlp_2(nz, 512*16*16, 512)
        self.forb3 = mlp_2(nz, 512*16*16, 512)

        self.fory4 = mlp_2(nz, 512*32*32, 512)
        self.forb4 = mlp_2(nz, 512*32*32, 512)




    def forward(self, z):
        w = self.z_to_w(z)
        y1 = self.fory1(w)
        b1 = self.forb1(w)
        y2 = self.fory2(w)
        b2 = self.forb2(w)
        y3 = self.fory3(w)
        b3 = self.forb3(w)
        y4 = self.fory4(w)
        b4 = self.forb4(w)

        
        y1 = y1.view(-1, 512, 4, 4)
        b1 = b1.view(-1, 512, 4, 4)
        y2 = y2.view(-1, 512, 8, 8)
        b2 = b2.view(-1, 512, 8, 8)
        y3 = y3.view(-1, 512, 16, 16)
        b3 = b3.view(-1, 512, 16, 16)
        y4 = y4.view(-1, 512, 32, 32)
        b4 = b4.view(-1, 512, 32, 32)

        # add these in the procees  latter.

        
        return self.main(input)



#         torch.Size([1, 512, 4, 4])
# torch.Size([1, 256, 8, 8])
# torch.Size([1, 128, 16, 16])
# torch.Size([1, 64, 32, 32])
# torch.Size([1, 3, 64, 64])