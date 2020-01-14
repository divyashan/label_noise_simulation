import numpy as np
import bisect
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def calibration_errors(aggregated_outputs, correct_labels):
    M = 10
    n = len(correct_labels)
    bin_boundaries = np.linspace(0, 1, M+1, endpoint=True)

    bins = [[] for i in range(M)]
    #Assign each output to a bin
    for i in range(len(aggregated_outputs)):
        prob_prediction = max(aggregated_outputs[i])
        j = bisect.bisect_left(bin_boundaries, prob_prediction)-1
        bins[j].append((aggregated_outputs[i], correct_labels[i]))

    expected_calibration_error = 0
    max_calibration_error = 0

    stats = {"bin_boundaries": bin_boundaries, 
             "accuracy": [],
             "confidence": []}

    for bin_number in range(len(bins)):
        confidence = bin_confidence(bins[bin_number], bin_number)
        accuracy = bin_accuracy(bins[bin_number])
        bin_size = len(bins[bin_number])
        expected_calibration_error += bin_size/n * abs(accuracy - confidence)
        max_calibration_error = max(max_calibration_error, abs(accuracy - confidence))
        stats["accuracy"].append(accuracy)
        stats["confidence"].append(confidence)
    stats["ECE"] = expected_calibration_error
    stats["MCE"] = max_calibration_error
    return stats

def bin_confidence(single_bin, bin_number):
    size = len(single_bin)
    if size == 0:
        return 0
    prob_prediction_sum = 0
    for sample in single_bin:
        aggregated_output, _ = sample
        prob_prediction_sum += aggregated_output[bin_number]
    return prob_prediction_sum/size

def bin_accuracy(single_bin):
    size = len(single_bin)
    if size == 0:
        return 0
    correct_samples = 0
    for sample in single_bin:
        aggregated_output, correct_label = sample
        if torch.argmax(aggregated_output) == correct_label:
            correct_samples += 1
    return correct_samples/size

def reliability_plot(stats, name):
    bin_boundaries = stats["bin_boundaries"][1:-1]
    accuracy = stats["accuracy"]

    plt.bar(bin_boundaries, accuracy, align="edge")
    xlocs, xlabs = plt.xticks()

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.xlim((0, 1))
    plt.title('Reliability Plot: {}'.format(name))
    plt.savefig('plots/reliability_plot/{}.pdf'.format(name))

