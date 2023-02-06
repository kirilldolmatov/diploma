import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
matplotlib.style.use('ggplot')


from utils.utils import create_noise, save_generator_image
from model.loss import bce_loss


class Trainer(object):

    def __init__(self, 
                device,
                generator,
                discriminator, 
                optim_g, 
                optim_d,
                epochs,
                sample_size,
                nz,
                train_size,
                train_loader
                ) -> None:
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.epochs = epochs
        self.sample_size = sample_size
        self.nz = nz
        self.train_size = train_size
        self.train_loade = train_loader


    # function to train the generator network
    def train_generator(self, optimizer, data_fake):
        b_size = data_fake.size(0)
        # get the real label vector
        real_label = torch.ones(b_size, 1).to(self.device)
        optimizer.zero_grad()
        # output by doing a forward pass of the fake data through discriminator
        output = self.discriminator(data_fake)
        loss = bce_loss(output, real_label)
        # compute gradients of loss
        loss.backward()
        # update generator parameters
        optimizer.step()
        return loss 


    # function to train the discriminator network
    def train_discriminator(self, optimizer, data_real, data_fake):
        b_size = data_real.size(0)
        # get the real label vector
        real_label = torch.ones(b_size, 1).to(self.device)
        # get the fake label vector
        fake_label = torch.zeros(b_size, 1).to(self.device)
        optimizer.zero_grad()
        # get the outputs by doing real data forward pass
        output_real = self.discriminator(data_real).view(-1)
        loss_real = bce_loss(output_real, real_label)
        # get the outputs by doing fake data forward pass
        output_fake = self.discriminator(data_fake)
        loss_fake = bce_loss(output_fake, fake_label)
        # compute gradients of real loss 
        loss_real.backward()
        # compute gradients of fake loss
        loss_fake.backward()
        # update discriminator parameters
        optimizer.step()
        return loss_real + loss_fake


    def train(self):
        losses_g = [] # to store generator loss after each epoch
        losses_d = [] # to store discriminator loss after each epoch

        self.generator.train()
        self.discriminator.train()

        # create the noise vector
        noise = create_noise(self.sample_size, self.nz)

        for epoch in range(self.epochs):
            loss_g = 0.0
            loss_d = 0.0
            total = int(len(self.train_size) / self.train_loader.batch_size)
            for bi, data in tqdm(enumerate(self.train_loader), total=total):
                image, _ = data
                image = image.to(self.device)
                b_size = len(image)
                # forward pass through generator to create fake data
                data_fake = self.generator(create_noise(b_size, self.nz)).detach()
                data_real = image
                loss_d += self.train_discriminator(self.optim_d, data_real, data_fake)
                data_fake = self.generator(create_noise(b_size, self.nz))
                loss_g += self.train_generator(self.optim_g, data_fake)

            # final forward pass through generator to create fake data...
            # ...after training for current epoch
            generated_img = self.generator(noise).cpu().detach()
            
            # save the generated torch tensor models to disk
            save_generator_image(generated_img, f"./outputs/gen_img{epoch}.png")
            epoch_loss_g = loss_g / bi # total generator loss for the epoch
            epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
            losses_g.append(epoch_loss_g)
            losses_d.append(epoch_loss_d)
            print(f"Epoch {epoch+1} of {self.epochs}")
            print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")
        
        print('DONE TRAINING')
        # save the model weights to disk
        torch.save(self.generator.state_dict(), './outputs/generator.pth')

        # plot and save the generator and discriminator loss
        plt.figure()
        plt.plot([x.detach().cpu().numpy() for x in losses_g], label='Generator loss')
        plt.plot([x.detach().cpu().numpy() for x in losses_d], label='Discriminator Loss')
        plt.legend()
        plt.savefig('./outputs/loss.png')
        plt.show()