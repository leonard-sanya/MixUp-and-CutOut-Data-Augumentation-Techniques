import torch # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data.sampler import SubsetRandomSampler # type: ignore
from torchvision import datasets, transforms # type: ignore
import torch.nn as nn # type: ignore

class CIFAR10Trainer:
    def __init__(self, batch_size, augment, mixup_alpha=0.4, num_epochs=30):
        self.batch_size = batch_size
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.num_epochs = num_epochs
        

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = torch.device("mps")
    
    def mixup_augmentation(self, x, y, alpha):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        idx = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[idx, :]
        y_a, y_b = y, y[idx]

        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def apply_mask(self, image, size=16, n_squares=1):
        b,c, h, w = image.shape
        new_image = image.clone()
        
        for _ in range(n_squares):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - size // 2, 0, h)
            y2 = np.clip(y + size // 2, 0, h)
            x1 = np.clip(x - size // 2, 0, w)
            x2 = np.clip(x + size // 2, 0, w)
            new_image[:, y1:y2, x1:x2] = 0

        return new_image
    
    def train(self, model, train_loader,valid_loader, optimizer, criterion, scheduler=None):
        print("Choose the type data augmentation technique to be applied?")
        type_augmentation = input(" 1 for MixUp and 2 for CutOut  ")
        print(" ")
        print(" ")
        model.to(self.device)
        model.train()
        
        ema_train_loss = None
        ema_valid_loss =None
        train_loss = []
        valid_loss = []
        
        train_acc = []
        valid_acc = []
        
        train_correct = 0
        valid_correct = 0
        train_total = 0
        valid_total = 0
        train_running_loss=0
        valid_running_loss=0

        total = 0
        print("------Training--------")
        for epoch in range(self.num_epochs):
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                train_total += labels.size(0)

                if type_augmentation == "1":
                    inputs, targets_a, targets_b, lam = self.mixup_augmentation(images, labels, self.mixup_alpha)
                    outputs = model(inputs)
                    output_idx = torch.argmax(outputs,dim =-1)
                    train_correct +=(labels==output_idx).sum().item()
                    optimizer.zero_grad()

                    loss_func = self.mixup_criterion(targets_a, targets_b, lam)
                    loss = loss_func(criterion, outputs)

                    train_running_loss += loss.item()*images.size(0)
                    loss.backward()
                    optimizer.step()

                    

                elif type_augmentation == "2":
                    masked_image = self.apply_mask(images, size=30, n_squares=10)
                    outputs = model(masked_image)

                    output_idx = torch.argmax(outputs,dim =-1)
                    train_correct +=(labels==output_idx).sum().item()
                    optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    train_running_loss += loss.item()*images.size(0)
                    loss.backward() 
                    optimizer.step()

                else:
                    print("Your choose is out of range")


            train_los = train_running_loss/train_total
            train_loss.append(train_los)

            train_ac = train_correct / train_total
            train_acc.append(train_ac)

            if scheduler is not None:
                scheduler.step()



              
            model.eval()    
            with torch.no_grad():
                for images, labels in valid_loader:
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    valid_total += labels.size(0)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    valid_running_loss += loss.item()*images.size(0)

                
                    # valid_loss.append(ema_valid_loss)
                    output_idx = torch.argmax(outputs,dim =-1)
                    valid_correct +=(labels==output_idx).sum().item()

                val_loss = valid_running_loss/valid_total
                valid_loss.append(val_loss)

                val_acc = valid_correct/valid_total
                valid_acc.append(val_acc)



            print(f"Epoch: {epoch+1}/{self.num_epochs},  Step: {i+1}/{len(train_loader)},  Train_loss: {train_los:.4f},  Train_Acc: {train_ac:.4f},  Val_loss: {val_loss:.4f},  Val_Acc: {val_acc:.4f}")

        print("Successfully finished traing")
        print(" ")
        return train_loss,train_acc,valid_loss,valid_acc
    
    
    def test(self, model, test_loader, criterion):
        model.to(self.device)
        model.eval()
        
        correct_top1 = 0
        correct_top5 = 0
        correct = 0
        total = 0

        print("------Evaluation--------")

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                
                _, pred_top1 = outputs.topk(1, dim=1, largest=True, sorted=True)
                correct_top1 += (pred_top1.view(-1) == labels).sum().item()
            
                _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
                correct_top5 += (pred_top5 == labels.view(-1, 1)).sum().item()

        top1_error = 1 - correct_top1 / total
        top5_error = 1 - correct_top5 / total
        accuracy = correct / total
        
        print(f"Top-1 Error: {top1_error * 100:.2f}%,  Top-5 Error: {top5_error * 100:.2f}%,  Test_Acc: {accuracy:.4f}%")
        
        return


        