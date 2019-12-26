import argparse
import torch
import yaml
import os
from modules import get_model
import math
import torchvision
import torchvision.transforms as transforms
from tta_dataset import TTADataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import numpy as np
DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/cifar_test.yaml"


def get_tta_dataset(dataset, augmentations, batch_size):
    transform = transforms.Compose([transforms.ToTensor()]) 

    if dataset =="CIFAR10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    elif dataset == "MNIST":
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testset = TTADataset(testset, augmentations)
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
    true_delta_matrices = config["test"]["true_delta_matrices"]
    num_classes = config["test"]["num_classes"]
    for matrix in true_delta_matrices:
        assert(len(matrix) == num_classes)
        for row in range(len(matrix)):
            assert(len(matrix[row]) == num_classes)
    batch_size = 1
    list_of_augmentations = config["data_loader"]["augmentations"]
    print("TTA AUGMENTATIONS: {}".format(list_of_augmentations))
    accuracy_given_auglist=[] 
    for augmentations in list_of_augmentations:
        val_loader = get_tta_dataset(config["data_loader"]["name"], augmentations, batch_size)
        writer_val = SummaryWriter(
        "runs/evaluate/validation/{}".format(",".join(augmentations)))
        accuracy_per_model = []
        for i in range(len(models_names)):
            model, num_classes = load_model(models_paths[i], device, config)
            acc, label_noise_matrix = run_evaluation(models_names[i], torch.tensor(true_delta_matrices[i]), augmentations, model, device, num_classes, val_loader, writer_val)
            generate_heatmap(torch.tensor(true_delta_matrices[i]), label_noise_matrix, models_names[i])
            accuracy_per_model.append(acc)
        accuracy_given_auglist.append(accuracy_per_model)
    plot_accuracy_with_diff_augmentations(list_of_augmentations, accuracy_given_auglist, models_names)

def plot_accuracy_with_diff_augmentations(list_of_augmentations, accuracy_given_auglist, models_names):

    for j in range(len(models_names)):
        objects = [", ".join(i) for i in list_of_augmentations]
        y_pos = np.arange(len(objects))
        performance = [accuracy_per_model[j] for accuracy_per_model in accuracy_given_auglist]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.xlabel('Augmentation')
        plt.title(models_names[j])
        plt.savefig('plots/aug_accuracy/{}.pdf'.format(j))

def run_evaluation(name_model, true_delta_matrix, augmentations, model, device, num_classes, val_loader, writer):
        mse = torch.nn.MSELoss(reduction = "sum")
        model.eval()
        eval_fraction = 1
        val_load_iter = iter(val_loader)
        counts = torch.zeros(num_classes)
        label_noise_matrix = torch.zeros(num_classes, num_classes)
        size = int(len(val_loader) * eval_fraction)
        accuracy = 0
        print("only use correct examples for label noise matrix")
        for i in range(size):
            data = val_load_iter.next()
            target_var = data["label"].float().to(device)
            output, pred_class = run_inference_on_augmentations(model, data, augmentations, num_classes, device, writer)
            if pred_class.item() == target_var.item():
                accuracy += 1
                # compute output
                label_noise_matrix[int(target_var.item())] += output 
                counts[int(target_var.item())] += 1.
           # measure accuracy and record loss
        for j in range(num_classes):
            label_noise_matrix[j] /= counts[j]

        
        mse_error = mse(label_noise_matrix, true_delta_matrix)
        kl_error = torch.zeros(len(label_noise_matrix))
        for i in range(len(label_noise_matrix)):
            kl_error[i] += torch.nn.functional.kl_div(label_noise_matrix[i],true_delta_matrix[i], 0)
        mean_kl = kl_error.mean()
        std_dev_kl = kl_error.std()
        print("Model: {}, MSE Error: {}, KL Error: {} +- {}, Acc: {}".format(name_model, mse_error, mean_kl, std_dev_kl, accuracy/size))
        return true_delta_matrix, label_noise_matrix

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

def run_inference_on_augmentations(model, data, augmentations, num_classes, device, writer):
    aggregated = torch.zeros(num_classes).float()
    num_total_augs = 0 
    softmax = torch.nn.Softmax(dim=1)

    for aug in augmentations:
        if  type(data[aug]) == list:
            for i in range(len(data[aug])):
               example = data[aug][i].float().to(device)
               output = softmax(model(example)).detach().reshape(num_classes).cpu()
               aggregated += output 
            num_total_augs += len(data[aug]) 
        else:
            example = data[aug].float().to(device)
            output = softmax(model(example)).detach().reshape(num_classes).cpu()
            aggregated += output
            num_total_augs += 1
   
    aggregated += softmax(model(data["image"].float().to(device))).detach().reshape(num_classes).cpu()
    writer.add_image("original", data["image"][0], 0) 

    final_output = aggregated/(num_total_augs + 1)
    return final_output, torch.argmax(final_output)



if __name__ == '__main__':
    main()
