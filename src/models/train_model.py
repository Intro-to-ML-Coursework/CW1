import numpy as np
import pickle


class DecisionTreeClassifier:
    """
    A Decision Tree classifier.

    This class implements a basic decision tree for classification. The tree is built using the
    decision tree learning algorithm, and it supports methods for training, prediction,
    saving the model to a file, and loading the model from a file.
    """

    def __init__(self):
        """
        Initialise the DecisionTreeClassifier instance.

        Attributes:
        decision_tree (dict): The dictionary representing the structure of the trained decision tree.
        depth (int): The depth of the decision tree.
        """
        self.decision_tree = dict()
        self.depth = 0

    def fit(self, training_dataset):
        """
        Train the decision tree classifier on the provided training dataset.

        Parameters:
        training_dataset (numpy.ndarray): The dataset used to train the decision tree. Each row corresponds to a
                                          sample, and the last column contains the labels.

        Returns:
        None
        """
        self.decision_tree, self.depth = decision_tree_learning(training_dataset, 0)

    def predict(self, x_test):
        """
        Predict labels for the provided test data.

        Parameters:
        x_test (numpy.ndarray): The test dataset (without labels). Each row corresponds to a sample.

        Returns:
        numpy.ndarray: An array of predicted labels for the test data.
        """
        predictions = np.zeros((len(x_test),))
        for i, sample in enumerate(x_test):
            # Start at the root of the decision tree
            node = self.decision_tree

            while node["attribute"] is not None:
                # Not a leaf, go to the left or right child
                if sample[node["attribute"]] < node["value"]:
                    node = node["left"]
                else:
                    node = node["right"]

            # Arrive at a leaf, make the prediction
            predictions[i] = node["value"]

        return predictions

    def save_model(self, path="decision_tree_model.pkl"):
        """
        Save the trained decision tree model to a file.

        Parameters:
        path (str, optional): The file path where the model should be saved (default is 'decision_tree_model.pkl').

        Returns:
        None
        """
        with open(path, "wb") as file:
            pickle.dump((self.decision_tree, self.depth), file)

    def load_model(self, path="decision_tree_model.pkl"):
        """
        Load a previously saved decision tree model from a file.

        Parameters:
        path (str, optional): The file path from where the model should be loaded
                              (default is 'decision_tree_model.pkl').

        Returns:
        None
        """
        with open(path, "rb") as file:
            (self.decision_tree, self.depth) = pickle.load(file)


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
    # Check if the training dataset is empty
    if training_dataset.size == 0:
        return dict(), -1

    # Retrieve unique labels and their respective counts
    unique_labels, counts = np.unique(training_dataset[:, -1], return_counts=True)
    if len(unique_labels) == 1:
        # All samples have the same label, return a leaf node
        return {"attribute": None, "value": unique_labels[0], "left": None, "right": None}, depth
    else:
        # Perform split that maximises information gain
        split_attribute, split_value, l_dataset, r_dataset = find_split(training_dataset)

        # Account for points with same features but different labels
        if l_dataset.size == 0 or r_dataset.size == 0:
            # Return a leaf node with the predicted label being the majority label
            return {"attribute": None, "value": unique_labels[np.argmax(counts)], "left": None, "right": None}, depth

        # Construct a new node
        node = {"attribute": split_attribute, "value": split_value}

        # Recursively build the left and right children
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
    # Calculate the entropy of the original dataset
    h_all = entropy(counts)

    # Iterate over each attribute and each value to determine the best split point
    max_gain, corresponding_attribute_index, corresponding_value = 0, None, None

    # Iterate over each attribute
    for attribute_index in range(training_dataset.shape[1] - 1):

        # Initialise the label counts of the left subset to be all zero
        left_label_counts = np.array(np.zeros_like(unique_labels, dtype=int))

        # Initialise the label counts of the right subset to be that of the original dataset
        right_label_counts = counts.copy()

        # Sort the dataset based on this attribute
        sorted_dataset = training_dataset[np.argsort(training_dataset[:, attribute_index])]
        sorted_indices = inverse_indices[np.argsort(training_dataset[:, attribute_index])]

        # Iterate over each value of this attribute, from smallest to largest
        previous_value = sorted_dataset[0, attribute_index]
        for i, label_index in enumerate(sorted_indices):
            if sorted_dataset[i, attribute_index] != previous_value:
                # Arrive at a new potential split point
                previous_value = sorted_dataset[i, attribute_index]

                # Calculate the information gain for this split point
                remainder = (i / num_samples * entropy(left_label_counts)
                             + (num_samples - i) / num_samples * entropy(right_label_counts))
                gain = h_all - remainder
                if gain > max_gain:
                    # Highest information gain so far, record the information gain, the attribute and the value of this
                    # split point
                    max_gain = gain
                    corresponding_attribute_index = attribute_index
                    # Use the mid-point for splitting
                    corresponding_value = np.mean(sorted_dataset[i-1:i+1, attribute_index])

            # Update the label counts for both left and right subset
            left_label_counts[label_index] += 1
            right_label_counts[label_index] -= 1

    if corresponding_attribute_index is None:
        return None, None, np.array([]), training_dataset
    # Partition the dataset into left and right subsets based on the split point that maximises information gain
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
    # Filter positive counts
    label_counts = label_counts[label_counts > 0]

    # Calculate proportions of each label
    proportions = label_counts / sum(label_counts)

    # Calculate the entropy
    return -np.sum(proportions * np.log2(proportions))


if __name__ == '__main__':

    # Load the dataset from file
    dataset_clean = np.array(np.loadtxt("../../wifi_db/clean_dataset.txt"))

    # Initialise the classifier
    decision_tree_classifier = DecisionTreeClassifier()

    # Train the classifier using the entire dataset
    decision_tree_classifier.fit(dataset_clean)

    # Some test samples
    x_test_samples = np.array([[-67, -61, -62, -67, -77, -83, -91], [-58, -57, -46, -55, -50, -87, -85]])

    # Make predictions
    predictions_of_samples = decision_tree_classifier.predict(x_test_samples)
    print(predictions_of_samples)

    # Save the trained model to a file
    decision_tree_classifier.save_model()

    # Load the trained model from a file
    decision_tree_classifier.load_model()

    # Examine the loaded decision tree model and its depth
    print(decision_tree_classifier.decision_tree)
    print(decision_tree_classifier.depth)
