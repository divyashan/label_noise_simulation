import argparse
import torch
import torch.optim as optim
import yaml
import os
import torchvision.models as models
import torch.nn as nn
from logistic_regression import LogisticRegression
from trainer import Trainer, get_dataset
from tensorboardX import SummaryWriter
from modules import get_model
DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/mnist_clean.yaml"


def main():
    """ Loads arguments and starts training."""
    parser = argparse.ArgumentParser(description="Simulated Noise Experiments")
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str)

    args = parser.parse_args()
    config_file = args.config
    # Load config
    assert os.path.exists(args.config), "Config file {} does not exist".format(
        args.config)

    with open(config_file) as fp:
        config = yaml.load(fp, yaml.Loader)

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    model_name = config["train"]["model"]
    num_classes = config["train"]["num_classes"]
    input_shape = config["data_loader"]["input_shape"]
    num_channels = input_shape[0]
    model = get_model(model_name, pretrained=False, num_channels=num_channels, num_classes=num_classes)
    model.to(device)
    print("Model name: {}".format(model_name))

    resume = config["train"]["resume"]
    if resume:
        if os.path.isfile(resume):
            print("Loading checkpoint {}".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint["epoch"]

            model.load_state_dict(checkpoint["state_dict"])

        else:
            start_epoch = 0
            print("No checkpoint found at {}".format(resume))

    else:
        start_epoch = 0
    batch_size = 32
    augmentations = config["data_loader"]["augmentations"]
    train_loader, val_loader  = get_dataset(config["data_loader"]["name"], config["data_loader"]["delta_matrix"], augmentations, input_shape, batch_size)

    if not os.path.exists(config["train"]["save_dir"]):
        os.makedirs(config["train"]["save_dir"])

    learning_rate = config["train"]["learning_rate"] or 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer_train = SummaryWriter(
        "runs/{}/training".format(config["train"]["save_as"]))
    writer_val = SummaryWriter(
        "runs/{}/validation".format(config["train"]["save_as"]))

    # Train the network
    num_epochs = config["train"]["num_epochs"]
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(device, model, writer_train, writer_val, train_loader, val_loader, criterion, optimizer)

    for epoch in range(start_epoch, num_epochs):
        trainer.train_epoch(epoch)
        save_checkpoint(
            {'epoch': epoch + 1, 'state_dict': model.state_dict()},
            filename=os.path.join(config["train"]["save_dir"],
                                  'checkpoint_{}.tar'.format(
                                      epoch))
        )

    print('Finished training')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()
