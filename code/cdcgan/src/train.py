import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.utils import save_image
import numpy as np
import matplotlib

from utils import save_generator_image, weights_init
from utils import label_fake, label_real, create_noise
from cdcgan import Generator, Discriminator
from dataset import BlastocystDataset

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

matplotlib.style.use('ggplot')


IMG_SIZE = 28 # we need to resize image
BATCH_SIZE = 32
LATENT_VECTOR_SIZE = 100 # latent vector size
BETA1 = 0.5 # beta1 value for Adam optimizer
LR = 0.0002 # learning rate according to paper
EPOCHS = 100 # number of epoch to train
Ksteps = 1 # number of discriminator steps for each generator step

INPUT_DATA = "/home/kirill/code/diploma/data/markup"
OUTPUT_DATA = "/home/kirill/code/diploma/data/outputs"

# set the computation device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])


image_path = Path(INPUT_DATA)
train_data = BlastocystDataset(targ_dir=image_path, transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) 

classes = train_data.classes
num_classes = len(classes)

# initialize models
generator = Generator(num_classes, LATENT_VECTOR_SIZE).to(DEVICE)
discriminator = Discriminator(num_classes).to(DEVICE)

# initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# loss function
criterion = nn.BCELoss()

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))


# labels for training images x for Discriminator training
labels_real = torch.ones((BATCH_SIZE, 1)).to(DEVICE)
# labels for generated images G(z) for Discriminator training
labels_fake = torch.zeros((BATCH_SIZE, 1)).to(DEVICE)
# Fix noise for testing generator and visualization
z_test = torch.randn(9, LATENT_VECTOR_SIZE).to(DEVICE)

# convert labels to onehot encoding
onehot = torch.zeros(num_classes, num_classes).scatter_(1, torch.arange(num_classes).view(num_classes, 1), 1)
# reshape labels to image size, with number of labels as channel
fill = torch.zeros([num_classes, num_classes, IMG_SIZE, IMG_SIZE])
# channel corresponding to label will be set one and all other zeros
for i in range(num_classes):
    fill[i, i, :, :] = 1
# create labels for testing generator
test_y = torch.concat([torch.arange(num_classes)] * num_classes).type(torch.LongTensor)
# convert to one hot encoding
test_Gy = onehot[test_y].to(DEVICE)


# List of values, which will be used for plotting purpose
D_losses = []
G_losses = []
Dx_values = []
DGz_values = []


# number of training steps done on discriminator 
step = 0
for epoch in range(EPOCHS):
  epoch_D_losses = []
  epoch_G_losses = []
  epoch_Dx = []
  epoch_DGz = []
  # iterate through data loader generator object
  for images, y_labels in train_loader:
    step += 1
    ############################
    # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # images will be send to gpu, if cuda available
    x = images.to(DEVICE)
    # preprocess labels for feeding as y input
    # D_y shape will be (batch_size, num_classes, 28, 28)
    D_y = fill[y_labels].to(DEVICE)
    # forward pass D(x)
    x_preds = discriminator(x, D_y)
    # calculate loss log(D(x))
    D_x_loss = criterion(x_preds, labels_real)
    
    # create latent vector z from normal distribution 
    z = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE).to(DEVICE)
    # create random y labels for generator
    y_gen = (torch.rand(BATCH_SIZE, 1)*num_classes).type(torch.LongTensor).squeeze()
    # convert genarator labels to onehot
    G_y = onehot[y_gen].to(DEVICE)
    # preprocess labels for feeding as y input in D
    # DG_y shape will be (batch_size, num_classes, 28, 28)
    DG_y = fill[y_gen].to(DEVICE)
    
    # generate image
    fake_image = generator(z, G_y)
    # calculate D(G(z)), fake or not
    z_preds = discriminator(fake_image.detach(), DG_y)
    # loss log(1 - D(G(z)))
    D_z_loss = criterion(z_preds, labels_fake)
    
    # total loss = log(D(x)) + log(1 - D(G(z)))
    D_loss = D_x_loss + D_z_loss
    
    # save values for plots
    epoch_D_losses.append(D_loss.item())
    epoch_Dx.append(x_preds.mean().item())
    
    # zero accumalted grads
    discriminator.zero_grad()
    # do backward pass
    D_loss.backward()
    # update discriminator model
    optim_d.step()
    
    ############################
    # Update G network: maximize log(D(G(z)))
    ###########################
        
    # if Ksteps of Discriminator training are done, update generator
    if step % Ksteps == 0:
      # As we done one step of discriminator, again calculate D(G(z))
      z_out = discriminator(fake_image, DG_y)
      # loss log(D(G(z)))
      G_loss = criterion(z_out, labels_real)
      # save values for plots
      epoch_G_losses.append(G_loss.item())
      epoch_DGz.append(z_out.mean().item())
      
      
      # zero accumalted grads
      generator.zero_grad()
      # do backward pass
      G_loss.backward()
      # update generator model
      optim_g.step()
  else:
    # calculate average value for one epoch
    D_losses.append(sum(epoch_D_losses)/len(epoch_D_losses))
    G_losses.append(sum(epoch_G_losses)/len(epoch_G_losses))
    Dx_values.append(sum(epoch_Dx)/len(epoch_Dx))
    DGz_values.append(sum(epoch_DGz)/len(epoch_DGz))
    
    print(f" Epoch {epoch+1}/{EPOCHS} Discriminator Loss {D_losses[-1]:.3f} Generator Loss {G_losses[-1]:.3f}"
         + f" D(x) {Dx_values[-1]:.3f} D(G(x)) {DGz_values[-1]:.3f}")
    
    # Generating images after each epoch and saving
    # set generator to evaluation mode
    generator.eval()
    with torch.no_grad():
      # forward pass of G and generated image
      fake_test = generator(z_test, test_Gy).cpu()
      # save images in grid of 10 * 10
      save_image(fake_test, f"{OUTPUT_DATA}/epoch_{epoch+1}.jpg", nrow=3, padding=0, normalize=True)
    # set generator to training mode
    generator.train()

plt.figure(figsize=(10,5))
plt.title("Discriminator and Generator loss during Training")
# plot Discriminator and generator loss
plt.plot(D_losses ,label="D Loss")
plt.plot(G_losses,label="G Loss")
# get plot axis
ax = plt.gca()
# remove right and top spine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# add labels and create legend
plt.xlabel("num_epochs")
plt.legend()
plt.show()