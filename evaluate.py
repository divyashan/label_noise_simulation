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
    for augmentations in list_of_augmentations:

        writer_val = SummaryWriter(
        "runs/evaluate/validation/{}".format(",".join(augmentations)))
        accuracy_per_model = []
        print(augmentations)
        kl_label_noise_per_model = {"mean": [], "std": []}
        for i in range(len(models_names)):
            model, num_classes = load_model(models_paths[i], device, config)
            val_loader = get_tta_dataset(config["data_loader"]["name"], augmentations, batch_size, model, device, hdf5s[i], num_classes)
            acc, label_noise_matrix, mean_kl, std_dev_kl  = run_evaluation(models_names[i], torch.tensor(true_delta_matrices[i]), augmentations, device, num_classes, val_loader, writer_val)
            accuracy_per_model.append(acc)
            kl_label_noise_per_model["mean"].append(mean_kl)
            kl_label_noise_per_model["std"].append(std_dev_kl)
        accuracy_given_auglist.append(accuracy_per_model)
        kl_given_auglist.append(kl_label_noise_per_model)
    plot_accuracy_with_diff_augmentations(list_of_augmentations, accuracy_given_auglist, models_names)
    plot_kl_with_diff_augmentations(list_of_augmentations, kl_given_auglist, models_names)

def plot_kl_with_diff_augmentations(list_of_augmentations, kl_dic, models_names):
    print(list_of_augmentations)
    print(kl_dic)
    print(models_names)
    for j in range(len(models_names)):
        objects = [", ".join(i) for i in list_of_augmentations]
        y_pos = np.arange(len(objects))
        performance_mean = [kl_per_model["mean"][j] for kl_per_model in kl_dic]
        performance_std = [kl_per_model["std"][j] for kl_per_model in kl_dic]

        plt.bar(y_pos, performance_mean, yerr=performance_std, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('KL Divergence')
        plt.xlabel('Augmentation')
        plt.title(models_names[j])
        print(models_names[j])
        xlocs, xlabs = plt.xticks()
        for i, v in enumerate(performance_mean):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
        plt.savefig('plots/aug_kl/{}.pdf'.format(models_names[j]))
        plt.close('all')


def plot_accuracy_with_diff_augmentations(list_of_augmentations, accuracy_given_auglist, models_names):
    print(list_of_augmentations)
    print(accuracy_given_auglist)
    print(models_names)
    for j in range(len(models_names)):
        objects = [", ".join(i) for i in list_of_augmentations]
        y_pos = np.arange(len(objects))
        performance = [accuracy_per_model[j] for accuracy_per_model in accuracy_given_auglist]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.xlabel('Augmentation')
        plt.title(models_names[j])
        print(models_names[j])
        xlocs, xlabs = plt.xticks()
        for i, v in enumerate(performance):
            plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
        plt.savefig('plots/aug_accuracy/{}.pdf'.format(models_names[j]))
        plt.close('all')

def run_evaluation(name_model, true_delta_matrix, augmentations, device, num_classes, val_loader, writer):
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

        expected_calibration_error, max_calibration_error = metrics.calibration_error(aggregated_outputs, correct_labels)
        mse_error = mse(label_noise_matrix, true_delta_matrix)
        kl_error = torch.zeros(len(label_noise_matrix))
        for i in range(len(label_noise_matrix)):
            kl_error[i] += torch.abs(torch.nn.functional.kl_div(label_noise_matrix[i],true_delta_matrix[i], reduction='sum'))
        mean_kl = kl_error.mean()
        std_dev_kl = kl_error.std()
        print("Model: {}, MSE Error: {}, KL Error: {} +- {}, ECE: {}, MCE, Acc: {}".format(name_model, mse_error, mean_kl, std_dev_kl, expected_calibration_error, max_calibration_error, accuracy/size))
        acc = accuracy/size
        return acc, label_noise_matrix, mean_kl, std_dev_kl

def generate_heatmap(true_delta_matrix, label_noise_matrix, name):
    fig, ax = plt.subplots(1, 6, gridspec_kw={'width_ratios':[1, 0.08, 1, 0.08, 1, 0.08]}, figsize=(26, 8))
    plt.title(name)
    g1 =sns.heatmap(torch.abs(label_noise_matrix - true_delta_matrix), cmap="YlGnBu", cbar=True, ax = ax[0], cbar_ax= ax[1], vmin=0, vmax=0.5)
    g2 =sns.heatmap(true_delta_matrix, cmap="YlGnBu", cbar=True, ax=ax[2], cbar_ax=ax[3], vmin=0, vmax=1)
    g3 =sns.heatmap(label_noise_matrix, cmap="YlGnBu", cbar=True, ax= ax[4], cbar_ax=ax[5], vmin=0, vmax=1)

    ax[0].set_title("Estimation Error")
    ax[2].set_title("True Label Noise")
    ax[4].set_title("Estimated Label Noise")
    for axis in ax:
        tly = axis.get_yticklabels()
        axis.set_yticklabels(tly, rotation=0)
    plt.savefig('heatmaps/aug_{}.png'.format(name))

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
