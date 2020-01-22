import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from noisy_dataset import NoisyDataset

def get_dataset(dataset, delta_matrix, augmentations,  batch_size):

    transform = []
    if "rot" in augmentations:
        transform.append(transforms.RandomRotation(degrees=5))
    if "five_crop" in augmentations:
        transform.append(transforms.RandomCrop(32, padding=4))
    if "hflip" in augmentations:
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    if dataset =="CIFAR10":
        transform.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform = transforms.Compose(transform)

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        tilde_trainset = NoisyDataset(trainset, delta_matrix)
        train_set, val_set = torch.utils.data.random_split(tilde_trainset, [int(len(trainset) * 0.95), len(trainset) - int(len(trainset) * 0.95) ])

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
   
    elif dataset == "MNIST":
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
        tilde_trainset = NoisyDataset(trainset, delta_matrix)
        train_set, val_set = torch.utils.data.random_split(trainset, [int(len(trainset) * 0.95), len(trainset) - int(len(trainset) * 0.95) ])

        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
 
    return trainloader, valloader,  tilde_trainset.class_weights

def get_dataset_for_rank_pruning(dataset, delta_matrix):
    transform = []
    transform.append(transforms.ToTensor())

    if dataset =="CIFAR10":
        transform.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        tilde_trainset = NoisyDataset(trainset, delta_matrix)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    elif dataset == "MNIST":
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
        tilde_trainset = NoisyDataset(trainset, delta_matrix)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
        #return images and noisy labels
    
    return tilde_trainset.original_dataset.data, tilde_trainset.corrupted_labels, testset.data, testset.targets

class Trainer(object):
    """ Trainer for Label Noise Experiments.

    Arguments:
        device (torch.device): The device on which the training will happen.

    """
    def __init__(self, device, model, writer_train, writer_val, train_loader, val_loader, criterion, optimizer):
        self._device = device
        self._model = model
        self._writer_train = writer_train
        self._writer_val = writer_val
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._criterion = criterion
        self._optimizer = optimizer


    def adjust_learning_rate(self):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2

    def train_epoch(self, epoch):
        """
            Method that trains the model for one epoch on the training set and
            reports losses to Tensorboard using the writer_train
        """
        self._model.train()
        for i, data in enumerate(self._train_loader):
            if i % 20 == 0:
                self.validate(i, epoch, 0.1)

                # switch to train mode
            input_var, target_var = data
            input_var = input_var.to(self._device)
            target_var = target_var.to(self._device)
 
            output = self._model(input_var)
            loss = self._criterion(output, target_var)

            # compute gradient and do optimization step
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
 
            if i + len(self._train_loader) * epoch % 1000 == 0:
                self.adjust_learning_rate()
            self._writer_train.add_scalar('data/loss', loss,
                                    i + len(self._train_loader) * epoch)
            print("Epoch: [{0}][{1}/{2}]\t Loss {loss:.4f} ".format(
                    epoch, i, len(self._train_loader), loss=loss))

    def validate(self, index=None, cur_epoch=None,eval_fraction=1):
        """

        Method that validates the model on the validation set and reports losses
        to Tensorboard using the writer

        """
        print("HELLO")
        epoch_length = len(self._train_loader)
        # switch to evaluate mode
        self._model.eval()
        loss_avg = 0.
        val_load_iter = iter(self._val_loader)
        for i in range(int(len(self._val_loader) * eval_fraction)):
            data = val_load_iter.next()
            input_var, target_var = data
            input_var = input_var.to(self._device)
            target_var = target_var.to(self._device)

                # compute output
            output = self._model(input_var)
            loss = self._criterion(output, target_var)
            loss_avg += loss
           # measure accuracy and record loss

        loss_avg /= int(len(self._val_loader) * eval_fraction)
        self._writer_val.add_scalar('data/loss', loss_avg,
                              cur_epoch * epoch_length + index)
        print('Test:[{0}][{1}/{2}]\tLoss {loss:.4f} '.format(
                    cur_epoch, index, epoch_length, loss=loss_avg))

