import numpy as np
from src.models.train_model import entropy


def information_gain(dataset, attribute, value):
    """
    Calculates the information gain from splitting a dataset based on a given attribute and value.

    Parameters:
    dataset (numpy.ndarray): The dataset to be split, where each row represents a data point
                             and the last column contains the labels.
    attribute (int): The index of the attribute (column) to split on.
    value (float): The threshold value for the split. Data points with the attribute value less than this threshold
                   go to the left subset, and others go to the right.

    Returns:
    float: The information gain, which represents the reduction in entropy due to the split.

    The function calculates the entropy of the original dataset, then splits it into two subsets (left and right)
    based on the given attribute and value. It calculates the weighted entropy of these two subsets and
    subtracts it from the original entropy to obtain the information gain.
    """
    left = dataset[dataset[:, attribute] < value]
    right = dataset[dataset[:, attribute] >= value]
    _, left_label_counts = np.unique(left[:, -1], return_counts=True)
    _, right_label_counts = np.unique(right[:, -1], return_counts=True)
    remainder = (len(left) / len(dataset) * entropy(left_label_counts)
                 + len(right) / len(dataset) * entropy(right_label_counts))
    _, total_label_counts = np.unique(dataset[:, -1], return_counts=True)
    return entropy(total_label_counts) - remainder


if __name__ == '__main__':
    dataset_clean = np.loadtxt("../../wifi_db/clean_dataset.txt")
    print(information_gain(dataset_clean, 0, -54))
