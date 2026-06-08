import torchvision
from torchvision import transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    train_data = EMNIST(root='./data',split='balanced',
                        train=True,download=True,
                        transform=transform)

    train_loader = DataLoader(train_data,batch_size=64,shuffle=True)

    test_data = EMNIST(root='./data',split='balanced',
                        train=False,download=True,
                        transform=transform)
    test_loader = DataLoader(test_data,batch_size=64)
    return train_loader, test_loader