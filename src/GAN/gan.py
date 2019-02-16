import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.GAN.discriminator import Discriminator as Dis
from src.GAN.generator import Generator as Gen
from src.utils.utils import timeit, imsave


class gan(object):
    def __init__(self, results_dir='../../results/', nz=100, image_size=64, epochs=5, lr=0.0002, beta1=0.5, n_gpu=1,
                 criterion=nn.BCELoss()):
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
        self.real_label = 1
        self.fake_label = 0

        self.n_gpu = n_gpu
        self.n_g_input = nz  # size of z latent vector (or gen input)
        self.image_size = image_size

        self.n_epochs = epochs
        self.beta1 = beta1
        self.lr = lr

        self.criterion = criterion

        # create networks
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.n_gpu > 0) else "cpu")
        print('device: ', self.device)
        self.net_g = Gen(n_gpu).to(self.device)
        self.net_d = Dis(n_gpu).to(self.device)

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        self.fixed_noise = torch.randn(image_size, self.n_g_input, 1, 1, device=self.device)

        self.data_loader = None
        self.optimizerG = None
        self.optimizerD = None

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

    def create_nets(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.n_gpu > 1):
            self.net_g = nn.DataParallel(self.net_g, list(range(self.n_gpu)))
            self.net_d = nn.DataParallel(self.net_d, list(range(self.n_gpu)))

        # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
        self.net_g.apply(weights_init)
        self.net_d.apply(weights_init)

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.net_d.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.net_g.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        print('Generator: ', self.net_g, sep='\n')
        print('Discriminator: ', self.net_d, sep='\n')

        return self

    @timeit
    def train(self):
        # Lists to keep track of progress
        G_losses = []
        D_losses = []

        print("Starting Training Loop")
        for epoch in range(self.n_epochs):
            for i, data in enumerate(self.data_loader, 0):  # foreach batch
                # ---------------------------- Discriminator ----------------------------

                # Train with all-real batch
                self.net_d.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, device=self.device)
                # forward, error, back
                output = self.net_d(real_cpu).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.n_g_input, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.net_g(noise)
                label.fill_(self.fake_label)
                # forward, error, back
                output = self.net_d(fake.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                # ---------------------------- Generator ----------------------------

                self.net_g.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                output = self.net_d(fake).view(-1)
                # Calculate G's loss based on D's output
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.n_epochs, i, len(self.data_loader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (i % 500 == 0) or ((epoch == self.n_epochs - 1) and (i == len(self.data_loader) - 1)):
                    with torch.no_grad():
                        fake = self.net_g(self.fixed_noise).detach().cpu()
                    title = self.results_dir + 'epoch_i ' + str(epoch) + '_' + str(i) + \
                            ' loss ' + str(round(errG.item() * 100) / 100)
                    imsave(fake, self.device, title)

        print('done training')

        # final stats
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.results_dir + 'final.jpg')

        # saving nets
        torch.save(self.net_d.state_dict(), '../../discriminator.tsr')
        torch.save(self.net_g.state_dict(), '../../generator.tsr')


if __name__ == '__main__':
    g = gan()
    g.load_data().create_nets()
    g.train()
