from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_cifar10(image_size, batch_size, root=r'/Users/kirill/code/diploma/code/dcgan/input/data'):
    transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # prepare the data
    train_data = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, len(train_data)