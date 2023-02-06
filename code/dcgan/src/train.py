import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim


from datasets import dataset
from model.model import Generator, Discriminator
from utils.utils import weights_init
from trainer.trainer import Trainer

# for mac m1 use 'mps', for other - 'cpu'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
# learning parameters / configurations according to paper
IMG_SIZE = 64 # we need to resize image to 64x64
BATCH_SIZE = 128
NZ = 100 # latent vector size
BETA1 = 0.5 # beta1 value for Adam optimizer
LR = 0.0002 # learning rate according to paper
SAMPLE_SIZE = 64 # fixed sample size
EPOCHS = 25 # number of epoch to train


def main():
    # get data
    train_loader, train_size = dataset.get_cifar10(IMG_SIZE, BATCH_SIZE)

    # initialize models
    generator = Generator(NZ).to(DEVICE)
    discriminator = Discriminator().to(DEVICE) 

    # initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # optimizers
    optim_g = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

    # train model
    trainer = Trainer(DEVICE,
                      generator,
                      discriminator, 
                      optim_g, 
                      optim_d,
                      EPOCHS,
                      SAMPLE_SIZE,
                      NZ,
                      train_size,
                      train_loader)

    trainer.train()

if __name__ == "__main__":
    main()


