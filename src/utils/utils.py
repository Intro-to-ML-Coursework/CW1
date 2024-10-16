import numpy as np
from src.models.train_model import entropy


def information_gain(dataset, attribute, value):
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
