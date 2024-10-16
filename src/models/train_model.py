import numpy as np


def decision_tree_learning(training_dataset, depth):
    unique_labels, counts = np.unique(training_dataset[:, -1], return_counts=True)
    if len(unique_labels) == 1:
        return {"attribute": None, "value": unique_labels[0], "left": None, "right": None}, depth
    else:
        split_attribute, split_value, l_dataset, r_dataset = find_split(training_dataset)
        if l_dataset.size == 0 or r_dataset.size == 0:
            return {"attribute": None, "value": unique_labels[np.argmax(counts)], "left": None, "right": None}, depth
        node = {"attribute": split_attribute, "value": split_value}
        node["left"], l_depth = decision_tree_learning(l_dataset, depth + 1)
        node["right"], r_depth = decision_tree_learning(r_dataset, depth + 1)
        return node, max(l_depth, r_depth)


def find_split(training_dataset):
    num_samples = len(training_dataset)
    unique_labels, inverse_indices, counts = np.unique(training_dataset[:, -1], return_inverse=True, return_counts=True)
    h_all = entropy(counts)
    max_gain, corresponding_attribute_index, corresponding_value = 0, None, None
    for attribute_index in range(training_dataset.shape[1] - 1):
        left_label_counts = np.zeros_like(unique_labels, dtype=int)
        right_label_counts = counts.copy()
        sorted_dataset = training_dataset[np.argsort(training_dataset[:, attribute_index])]
        sorted_indices = inverse_indices[np.argsort(training_dataset[:, attribute_index])]
        previous_value = sorted_dataset[0, attribute_index]
        for i, label_index in enumerate(sorted_indices):
            if sorted_dataset[i, attribute_index] != previous_value:
                previous_value = sorted_dataset[i, attribute_index]
                remainder = (i / num_samples * entropy(left_label_counts)
                             + (num_samples - i) / num_samples * entropy(right_label_counts))
                gain = h_all - remainder
                if gain > max_gain:
                    max_gain = gain
                    corresponding_attribute_index = attribute_index
                    corresponding_value = sorted_dataset[i, attribute_index]
            left_label_counts[label_index] += 1
            right_label_counts[label_index] -= 1
    l_dataset = training_dataset[training_dataset[:, corresponding_attribute_index] < corresponding_value]
    r_dataset = training_dataset[training_dataset[:, corresponding_attribute_index] >= corresponding_value]
    return corresponding_attribute_index, corresponding_value, l_dataset, r_dataset


def entropy(label_counts):
    label_counts = label_counts[label_counts > 0]
    proportions = label_counts / sum(label_counts)
    return -np.sum(proportions * np.log2(proportions))


if __name__ == '__main__':
    dataset_clean = np.loadtxt("../../wifi_db/clean_dataset.txt")
    dataset_noisy = np.loadtxt("../../wifi_db/noisy_dataset.txt")
    print(dataset_clean)
    print(dataset_noisy)
    print(decision_tree_learning(dataset_clean, 0))
    print(decision_tree_learning(dataset_noisy, 0))
