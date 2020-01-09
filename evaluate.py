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
plt.switch_backend('agg')
import seaborn as sns
import numpy as np
import metrics
import pandas as pd

DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/cifar_test.yaml"


def get_tta_dataset(dataset, augmentations, batch_size, model, device, hdf5, num_classes):
    transform = transforms.Compose([transforms.ToTensor()]) 

    if dataset =="CIFAR10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    elif dataset == "MNIST":
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testset = TTADataset(testset, augmentations)
    tta_hdf5_dataset = TTA_HDF5_Dataset(model, testset, device, hdf5, num_classes) 
    testloader = torch.utils.data.DataLoader(tta_hdf5_dataset, batch_size=batch_size,
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
    true_delta_matrices = config["test"]["true_delta_matrices"]
    num_classes = config["test"]["num_classes"]
    hdf5s = config["data_loader"]["hdf5"]

    for matrix in true_delta_matrices:
        assert(len(matrix) == num_classes)
        for row in range(len(matrix)):
            assert(len(matrix[row]) == num_classes)
    batch_size = 1
    list_of_augmentations = config["data_loader"]["augmentations"]
    print("TTA AUGMENTATIONS: {}".format(list_of_augmentations))
    accuracy_given_auglist=[] 
    kl_given_auglist = []

    data = []
    for augmentations in list_of_augmentations:
        for i in range(len(models_names)):
            val_loader = get_tta_dataset(config["data_loader"]["name"], augmentations, batch_size, None, device, hdf5s[i], num_classes)
            stats = run_evaluation(models_names[i], torch.tensor(true_delta_matrices[i]), augmentations, device, num_classes, val_loader)
            data.append(stats)

    df = pd.DataFrame(data)
    df.to_csv("results.csv", index=False)
#    plot_accuracy_with_diff_augmentations(list_of_augmentations, accuracy_given_auglist, models_names)
#    plot_kl_with_diff_augmentations(list_of_augmentations, kl_given_auglist, models_names)


def run_evaluation(name_model, true_delta_matrix, augmentations, device, num_classes, val_loader):
        mse = torch.nn.MSELoss(reduction = "sum")
        eval_fraction = 1
        val_load_iter = iter(val_loader)
        counts = torch.zeros(num_classes)
        label_noise_matrix = torch.zeros(num_classes, num_classes)
        size = int(len(val_loader) * eval_fraction)
        accuracy = 0
        print("only use correct examples for label noise matrix")

        aggregated_outputs = []
        correct_labels = []
        for i in range(size):
            data = val_load_iter.next()
            target_var = data["label"].float().to(device)
            output, pred_class = run_inference_on_augmentations(data, augmentations)
            aggregated_outputs.append(output.detach())
            correct_labels.append(target_var.item())
            if pred_class.item() == target_var.item():
                accuracy += 1
                # compute output
                label_noise_matrix[int(target_var.item())] += output 
                counts[int(target_var.item())] += 1.
           # measure accuracy and record loss
        for j in range(num_classes):
            label_noise_matrix[j] /= counts[j]

        expected_calibration_error, max_calibration_error = metrics.calibration_errors(aggregated_outputs, correct_labels)
        mse_error = mse(label_noise_matrix, true_delta_matrix)
        kl_error = torch.zeros(len(label_noise_matrix))
        for i in range(len(label_noise_matrix)):
            kl_error[i] += torch.abs(torch.nn.functional.kl_div(label_noise_matrix[i],true_delta_matrix[i], reduction='sum'))
        mean_kl = kl_error.mean()
        std_dev_kl = kl_error.std()
        print("Model: {}, MSE Error: {}, KL Error: {} +- {}, ECE: {}, MCE: {}, Acc: {}".format(name_model, mse_error, mean_kl, std_dev_kl, expected_calibration_error, max_calibration_error, accuracy/size))
        acc = accuracy/size
        stats = {"Accuracy": acc,
#                 "Label Noise Matrix": label_noise_matrix,
                 "KL Mean": mean_kl.item(),
                 "KL Std": std_dev_kl.item(),
                 "MSE": mse_error.item(),
                 "ECE": expected_calibration_error.item(),
                 "MCE": max_calibration_error.item(),
                 "Augmentations": augmentations,
                 "Model Name": name_model}
        return stats

def run_inference_on_augmentations(data, augmentations):
    num_classes = data["image_output"].shape[2]
    aggregated = torch.zeros(num_classes).float()
    num_total_augs = 0 
    softmax = torch.nn.Softmax(dim=1)

    for aug in augmentations:
        if not aug.startswith("rot_"):
            aggregated += torch.sum(data["{}_output".format(aug)], 1).unsqueeze(dim=0).reshape(num_classes)
            num_total_augs += data["{}_output".format(aug)].shape[0] 
        else:
            rot_aug = get_key_that_starts_with_rot(data)
            n_rotations = int(aug.split("_")[1])
            aggregated += torch.sum(data[rot_aug][:n_rotations, :], 1).unsqueeze(dim=0).reshape(num_classes)
            num_total_augs += n_rotations 

    aggregated += data["image_output"].unsqueeze(dim=0).reshape(num_classes)
    aggregated = aggregated/(num_total_augs+1)
    final_output = softmax(aggregated.unsqueeze(dim=0)).detach().reshape(num_classes).cpu()
    return final_output, torch.argmax(final_output)

def get_key_that_starts_with_rot(data):
    for key in data.keys():
        if key.startswith("rot_"):
            return key

if __name__ == '__main__':
    main()
