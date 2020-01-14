from metrics import reliability_plot
import torch
import numpy as np


def test_generate_heatmap():
    true_delta_matrix = torch.eye(10)
    label_noise_matrix = torch.eye(10) * 0.9
    name = "test"

    generate_heatmap(true_delta_matrix, label_noise_matrix, name)

def test_reliability_plot():
    bin_boundaries = np.linspace(0, 1, 11, endpoint=True)
    print(bin_boundaries)
    accuracy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    stats = {'accuracy': accuracy, 'bin_boundaries': bin_boundaries}
    reliability_plot(stats, "Test Plot")

test_reliability_plot()
