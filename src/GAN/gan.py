# from __future__ import print_function
# %matplotlib inline
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

from src.GAN.discriminator import Discriminator as Dis
from src.GAN.generator import Generator as Gen
from src.utils.utils import timeit


class gan(object):
    def __init__(self, results_dir='../../results/', image_size=64, epochs=5, lr=0.0002, beta1=0.5, n_gpu=1):
        """
        GAN object containing a discriminator and generator
        :param results_dir: folder for the results
        :param image_size: spatial size of training images
        :param epochs: number of training epochs
        :param lr: learning rate
        :param beta1: optimiser hyper-param
        :param n_gpu: number of GPUs
        """
        self.seed = 999
        self.results_dir = results_dir

        self.n_g_input = 100  # size of z latent vector (or gen input)
        self.image_size = image_size

        self.n_epochs = epochs
        self.beta1 = beta1
        self.lr = lr

        self.net_g = Gen(n_gpu)
        self.net_d = Dis(n_gpu)

        self.data_loader = None

        device = torch.device("cuda:0" if (torch.cuda.is_available() and self.n_gpu > 0) else "cpu")
        print('device: ', device)

    def load_data(self, data_dir='../../data/celeba', batch_size=128, workers=2):
        dataset = dset.ImageFolder(root=data_dir,
                                   transform=transforms.Compose([
                                       transforms.Resize(self.image_size),
                                       transforms.CenterCrop(self.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Create the data loader
        self.data_loader = \
            torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

        return self


@timeit
def main():
    # Set random seem for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Root directory for dataset
    data_dir = "../../data/celeba"
    # Prefix of where to save img results
    results_dir = '../../results/'

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of training epochs
    num_epochs = 10

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print('device: ', device)

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Create the generator
    net_g = Gen(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net_g = nn.DataParallel(net_g, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    net_g.apply(weights_init)

    # Create the Discriminator
    net_d = Dis(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        net_d = nn.DataParallel(net_d, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    net_d.apply(weights_init)

    print('Generator: ', net_g, sep='\n')
    print('Discriminator: ', net_d, sep='\n')

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            net_d.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = net_d(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = net_g(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = net_d(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            net_g.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = net_d(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = net_g(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                plt.figure(figsize=(8, 8))
                plt.title('Imgs iter')
                plt.imshow(
                    np.transpose(img_list[len(img_list) - 1].to(device)[:64].cpu().numpy(), (1, 2, 0)))
                plt.savefig(
                    results_dir + 'epoch_i ' + str(epoch) + '_' + str(i) + ' loss ' +
                    str(round(errG.item() * 100) / 100) + '.jpg')

            iters += 1

    print('done training')

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(results_dir + 'final.jpg')

    torch.save(net_d.state_dict(), '../../discriminator.tsr')
    torch.save(net_g.state_dict(), '../../generator.tsr')


if __name__ == '__main__':
    main()
