import argparse
import torch
import yaml
import os
from modules import get_model
import math
import torchvision
import torchvision.transforms as transforms
from data_loaders import TTADataset, TTA_HDF5_Dataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from trainer import get_dataset
plt.switch_backend('agg')
import seaborn as sns
import numpy as np
import metrics
import pandas as pd
import skorch
from cleanlab.classification import LearningWithNoisyLabels
torch.manual_seed(0)
DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/cifar_test.yaml"


def get_tta_dataset(dataset, augmentations, batch_size, model, device, hdf5, num_classes):
    transform = transforms.ToTensor()
    if dataset =="CIFAR10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    elif dataset == "MNIST":
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

    return testloader


def load_model(model_path, device, config):
    model_name = config["test"]["model"]
    num_classes = config["test"]["num_classes"]
    num_channels = config["data_loader"]["input_shape"][0]
    model = get_model(model_name, pretrained=False, num_channels=num_channels, num_classes=num_classes)
    model.to(device)
    print("Loading {} into {}".format(model_path, model_name))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    return model, num_classes

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

    device = torch.device(config["test"]["device"] if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    models_names = config["test"]["models_names"]
    models_paths = config["test"]["models_paths"]
    num_classes = config["test"]["num_classes"]

    batch_size = 1
    data = []
    list_of_augmentations = config["data_loader"]["augmentations"]
 
    for augmentations in list_of_augmentations:
        for i in range(len(models_names)):
            model, _ = load_model(models_paths[i], device, config)
            test_loader = get_tta_dataset(config["data_loader"]["name"], augmentations, batch_size, model, device, hdf5s[i], num_classes)
            learn_transform(model, models_names[i], device, num_classes, test_loader)


def learn_rot_tf(model, model_name, device, num_classes, test_loader):
     ## We are interested in learning whether we can recommend the correct augmentation
     ## at test time from the outputs of a model.

     softmax = torch.nn.Softmax(dim=0)
     for i, data in enumerate(test_loader):
         corrupted_rotation_angle = random.randint(-10, 10)
         corrupted_rot_transform = transforms.RandomRotation((corrupted_rotation_angle-0.5, corrupted_rotation_angle +0.5))
         img, label = data
         corrupted_img = rot_transform(img)

         rot_angles = [i in range(-10, 10)]

         maximum_softmax_output = []

         for rot in rot_angles:
             rot_tf = transforms.RandomRotation((rot-0.5, rot +0.5))
             output = softmax(model(rot_tf(corrupted_img)).squeeze().squeeze())
             maximum_softmax_output.append(max(output))
         
         plot_stability(rot_angles, maximum_softmax_output, corrupted_rotation_angle, model_name)

def plot_stability(rot_angles, maximum_softmax_output, corrupted_angle, model_name):

    fig, ax = plt.figure(figsize=(10, 10))

    ax.plot(rot_angles, maximum_softmax_output, label="Max Softmax")
    ax.plot([corrupted_angle, corrupted_angle], [0, 1], label="Corruption Angle")
    ax.set_title("Test-Time Rotation Angle vs. Maximum Softmax Accuracy")
    plt.savefig("{}.pdf".format(model_name))
