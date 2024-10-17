import numpy as np


def decision_tree_learning(training_dataset, depth):
    """
    Recursively builds a decision tree using the training dataset and tracks its depth.

    Parameters:
    training_dataset (numpy.ndarray): The dataset used to build the decision tree.
                                       Each row represents a data point, and the last column contains labels.
    depth (int): The current depth of the tree during the recursion. It tracks how deep the tree is.

    Returns:
    tuple:
        - dict: A dictionary representing the current node of the decision tree. It contains:
            - 'attribute' (int or None if leaf): The index of the attribute used to split the data at this node.
            - 'value' (float): The value of the split at this node or the class label for leaf nodes.
            - 'left' (dict): The left subtree (or None if leaf).
            - 'right' (dict): The right subtree (or None if leaf).
        - int: The depth of the tree from this node downwards.

    If all examples in the current dataset have the same label, the function returns a leaf node.
    """
    unique_labels, counts = np.unique(training_dataset[:, -1], return_counts=True)
    if len(unique_labels) == 1:
        return {"attribute": None, "value": unique_labels[0], "left": None, "right": None}, depth
    else:
        split_attribute, split_value, l_dataset, r_dataset = find_split(training_dataset)
        node = {"attribute": split_attribute, "value": split_value}
        node["left"], l_depth = decision_tree_learning(l_dataset, depth + 1)
        node["right"], r_depth = decision_tree_learning(r_dataset, depth + 1)
        return node, max(l_depth, r_depth)


def find_split(training_dataset):
    """
    Finds the best attribute and value to split the dataset for decision tree learning based on information gain.

    Parameters:
    training_dataset (numpy.ndarray): The dataset to be split. Each row represents a data point,
                                      and the last column contains labels.

    Returns:
    tuple:
        - int: The index of the best attribute to split on.
        - float: The value of the best attribute to split the dataset.
        - numpy.ndarray: The left subset of the dataset where the attribute's value is less than the split value.
        - numpy.ndarray: The right subset of the dataset where the attribute's value is greater than
          or equal to the split value.

    The function computes the entropy of the entire dataset and attempts to split the dataset by each attribute.
    It evaluates each possible split by sorting the dataset by attribute values, and for each unique value,
    it calculates the information gain. The split that maximises information gain is selected.
    """
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
    """
    Computes the entropy of a set of labels.

    Parameters:
    label_counts (numpy.ndarray): An array representing the counts of each unique label in the dataset.

    Returns:
    float: The entropy value, representing the uncertainty or impurity of the label distribution.

    The function first removes any zero counts from the label distribution, then calculates the proportion
    of each label. It uses the formula for entropy: H = -Σ(p * log2(p)), where p is the proportion of each label.
    """
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
