import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data.sampler import SubsetRandomSampler # type: ignore
from torchvision import datasets, transforms # type: ignore
import torch.nn as nn # type: ignore
import matplotlib.pyplot as plt

from model import AlexNet
from data import DataLoader
from train import CIFAR10Trainer

print("Choose the type data to be used?")
type = input(" 10 for CIFAR10 or 100 for CIFAR100  ")

data_loader = DataLoader(
    Cifar_type =type,
    data_dir='./data',
    batch_size=64,
    augment=False,
    random_seed=1,
    valid_size=0.1
)

train_loader, valid_loader = data_loader.train_loader, data_loader.valid_loader
test_loader = data_loader.test_loader

num_classes = 10 if type == "10" else 100
num_epochs = 2
batch_size = 64
augment = True
mixup_alpha = 0.4
learning_rate = 0.01

if torch.cuda.is_available():
    device = "cuda"
else:
    device = torch.device("mps")


model_1 = AlexNet(num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model_1.parameters(), lr=learning_rate, weight_decay=5e-3, momentum = 0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.1)

# Train the model
total_step = len(train_loader)


# Example usage

trainer = CIFAR10Trainer(batch_size, augment, mixup_alpha, num_epochs)
def main():
    # Train the model
    train_loss,train_acc,valid_loss,valid_acc = trainer.train(model_1, train_loader,valid_loader, optimizer, criterion)
    # train_loss,train_acc = trainer.train(model_1, train_loader,valid_loader, optimizer, criterion)

    trainer.test(model_1,test_loader,criterion)
    print(" ")
    print(" ")
    # Plotting the training and validation loss
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses')
    plt.legend()
    plt.show()
    
    print(" ")
    print(" ")

    # Plotting the training and validation loss
    plt.plot(train_acc, label='Train Acc')
    plt.plot(valid_acc, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracies')
    plt.legend()
    plt.show()

         
if __name__ == "__main__":
    main()