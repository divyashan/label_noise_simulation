import torch
import torch.nn as nn


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
                self.validate(i, epoch, 0.01)

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

        epoch_length = len(self._train_loader)
        # switch to evaluate mode
        self._model.eval()

        val_load_iter = iter(self._val_loader)
        for i in range(int(len(self._val_loader) * eval_fraction)):
            data = val_load_iter.next()
            input_var, target_var = data
            input_var = input_var.to(self._device)
            target_var = target_var.to(self._device)
                # compute output
            output = self._model(input_var)
            loss = self._criterion(output, target_var)

           # measure accuracy and record loss

        if index is not None:
            self._writer_val.add_scalar('data/loss', loss,
                              cur_epoch * epoch_length + index)
            print('Test:[{0}][{1}/{2}]\tLoss {loss:.4f} '.format(
                    cur_epoch, index, epoch_length, loss=loss))
