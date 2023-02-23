import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib

from utils import save_generator_image, weights_init
from utils import label_fake, label_real, create_noise
from dcgan import Generator, Discriminator
from dataset import BlastocystDataset

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

matplotlib.style.use('ggplot')

# learning parameters / configurations according to paper
image_size = 64 # we need to resize image to 64x64
batch_size = 128
nz = 1000 # latent vector size
beta1 = 0.5 # beta1 value for Adam optimizer
lr = 0.003 # learning rate according to paper
sample_size = 1 # fixed sample size
epochs = 300 # number of epoch to train

INPUT_DATA = '/home/kirill/code/diploma/data/input'
OUTPUT_DATA = '/home/kirill/code/diploma/data/outputs'

# set the computation device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# image transforms
# transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), 
#     (0.5, 0.5, 0.5)),
# ])

transform_blastocyst = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])

# prepare the data
# train_data = datasets.CIFAR10(
#     root=INPUT_DATA,
#     train=True,
#     download=False,
#     transform=transform
# )
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

image_path = Path("/home/kirill/code/diploma/data/markup")

train_data = BlastocystDataset(targ_dir=image_path, transform=transform_blastocyst)

train_loader = DataLoader(dataset=train_data, 
                              batch_size=batch_size,
                              shuffle=True) 
# initialize models
generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)


# initialize generator weights
generator.apply(weights_init)
# initialize discriminator weights
discriminator.apply(weights_init)

print('##### GENERATOR #####')
print(generator)
print('######################')

print('\n##### DISCRIMINATOR #####')
print(discriminator)
print('######################')

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# loss function
criterion = nn.BCELoss()

losses_g = [] # to store generator loss after each epoch
losses_d = [] # to store discriminator loss after each epoch

# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    # get the real label vector
    real_label = label_real(b_size)
    # get the fake label vector
    fake_label = label_fake(b_size)

    optimizer.zero_grad()

    # get the outputs by doing real data forward pass
    output_real = discriminator(data_real).view(-1)
    loss_real = criterion(output_real, real_label.squeeze())

    # get the outputs by doing fake data forward pass
    output_fake = discriminator(data_fake).view(-1)
    loss_fake = criterion(output_fake, fake_label.squeeze())

    # compute gradients of real loss 
    loss_real.backward()
    # compute gradients of fake loss
    loss_fake.backward()
    # update discriminator parameters
    optimizer.step()

    return loss_real + loss_fake

# function to train the generator network
def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)
    # get the real label vector
    real_label = label_real(b_size)

    optimizer.zero_grad()

    # output by doing a forward pass of the fake data through discriminator
    output = discriminator(data_fake).view(-1)
    loss = criterion(output, real_label.squeeze())

    # compute gradients of loss
    loss.backward()
    # update generator parameters
    optimizer.step()

    return loss    

# create the noise vector
noise = create_noise(sample_size, nz)
# print('SIZE', noise.size())
# print('NOISE', noise)

generator.train()
discriminator.train()

for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
        image, _ = data
        image = image.to(device)
        b_size = len(image)
        # forward pass through generator to create fake data
        data_fake = generator(create_noise(b_size, nz)).detach()
        data_real = image
        loss_d += train_discriminator(optim_d, data_real, data_fake)
        data_fake = generator(create_noise(b_size, nz))
        loss_g += train_generator(optim_g, data_fake)

    # final forward pass through generator to create fake data...
    # ...after training for current epoch
    generated_img = generator(noise).cpu().detach()
    # save the generated torch tensor models to disk
    save_generator_image(generated_img, f"{OUTPUT_DATA}/gen_img{epoch}.png")
    epoch_loss_g = loss_g / bi # total generator loss for the epoch
    epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)

    print(f"Epoch {epoch+1} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

print('DONE TRAINING')
# save the model weights to disk
torch.save(generator.state_dict(), f'{OUTPUT_DATA}/generator.pth')

# plot and save the generator and discriminator loss
plt.figure()
plt.plot([x.detach().cpu().numpy() for x in losses_g], label='Generator loss')
plt.plot([x.detach().cpu().numpy() for x in losses_d], label='Discriminator Loss')
plt.legend()
plt.savefig(f'{OUTPUT_DATA}/loss.png')
plt.show()