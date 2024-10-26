import numpy as np
from numpy.random import default_rng
from src.models.train_model import DecisionTreeClassifier


def confusion_matrix(y_gold, y_pred, class_labels=None):

    """
    Calculate the confusion matrix for a classification problem.

    Parameters:
    y_gold (np.ndarray): The correct ground truth/gold standard labels.
    y_pred (np.ndarray): The predicted labels.
    class_labels (np.ndarray): A list of unique class labels. Defaults to the union of y_gold and y_pred.

    Returns:
    np.ndarray: Shape (C, C), where C is the number of classes.
                Rows are ground truth per class, columns are predictions.
    """

    # If no class_labels are given, we obtain the set of unique class labels from the union of
    # the ground truth annotation and the prediction
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y_gold, y_pred)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    # For each correct class (row),
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # Get predictions where the ground truth is the current class label
        indices = (y_gold == label)
        predictions = y_pred[indices]

        # Get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # Convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # Fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


def accuracy(confusion):
    acc = 0
    if np.sum(confusion) > 0:
        acc = np.sum(np.diag(confusion)) / np.sum(confusion)
    return acc


def precision(confusion):
    precisions = np.zeros((len(confusion),))

    for i in range(confusion.shape[1]):
        if np.sum(confusion[:, i]) > 0:
            precisions[i] = confusion[i, i] / np.sum(confusion[:, i])

    return precisions


def recall(confusion):
    recalls = np.zeros((len(confusion),))

    for i in range(confusion.shape[0]):
        if np.sum(confusion[i, :]) > 0:
            recalls[i] = confusion[i, i] / np.sum(confusion[i, :])

    return recalls


def f1_measure(precisions, recalls):
    f1_measures = np.zeros((len(precisions),))

    for i, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f1_measures[i] = 2 * p * r / (p + r)

    return f1_measures


def calculate_metrics(confusion):

    """
    Calculate the accuracy, precisions, recalls, and F1-measures from the confusion matrix.

    Parameters:
    conf_matrix (np.ndarray): The confusion matrix.

    Returns:
    float: The accuracy.
    np.ndarray: The precision for each class.
    np.ndarray: The recall for each class.
    np.ndarray: The F1-measure for each class.
    """

    acc = accuracy(confusion)
    recalls = recall(confusion)
    precisions = precision(confusion)
    f1_measures = f1_measure(precisions, recalls)

    return acc, recalls, precisions, f1_measures


def cross_validation(dataset, classifier, num_folds=10, random_generator=default_rng()):

    """
    Perform k-fold cross-validation to evaluate the classifier.

    Parameters:
    dataset (np.ndarray): The dataset containing all data.
    classifier (object): The classifier to evaluate.
    num_folds (int): The number of folds.

    Returns:
    np.ndarray: The average confusion matrix.
    float: The average accuracy.
    np.ndarray: The average precision for each class.
    np.ndarray: The average recall for each class.
    np.ndarray: The average F1 measure for each class.
    """

    shuffled_indices = random_generator.permutation(dataset.shape[0])
    split_indices = np.array_split(shuffled_indices, num_folds)
    unique_labels = np.unique(dataset[:, -1])

    cumulative_conf_matrix = np.zeros((len(unique_labels), len(unique_labels)))
    cumulative_accuracy = 0.
    cumulative_recalls = np.zeros((len(unique_labels),))
    cumulative_precisions = np.zeros((len(unique_labels),))
    cumulative_f1_measures = np.zeros((len(unique_labels),))

    for i in range(num_folds):
        # Split the dataset into train and test sets for this fold
        test_indices = split_indices[i]
        train_indices = np.concatenate(split_indices[:i] + split_indices[i+1:])
        training_set = dataset[train_indices]
        x_test = dataset[test_indices, :-1]
        y_test = dataset[test_indices, -1]

        # Train the model
        classifier.fit(training_set)

        # Predict the labels
        predictions = classifier.predict(x_test)

        # Get metrics for this fold
        confusion = confusion_matrix(y_test, predictions, unique_labels)
        acc, recalls, precisions, f1_measures = calculate_metrics(confusion)

        # Accumulate the metrics for calculating the average
        cumulative_conf_matrix += confusion
        cumulative_accuracy += acc
        cumulative_recalls += recalls
        cumulative_precisions += precisions
        cumulative_f1_measures += f1_measures

    # Calculate average metrics
    avg_confusion = cumulative_conf_matrix / num_folds
    avg_accuracy = cumulative_accuracy / num_folds
    avg_recall = cumulative_recalls / num_folds
    avg_precision = cumulative_precisions / num_folds
    avg_f1_measure = cumulative_f1_measures / num_folds

    # Print the average metrics to the console
    print("Average confusion matrix:")
    print(avg_confusion)
    print("Average accuracy:")
    print(avg_accuracy)
    print("Class labels:")
    print(unique_labels)
    print("Average recall for each class:")
    print(avg_recall)
    print("Average precision for each class:")
    print(avg_precision)
    print("Average F1-measure for each class:")
    print(avg_f1_measure)


def evaluate(test_db, trained_tree):
    x_test = test_db[:, :-1]
    y_test = test_db[:, -1]

    predictions = trained_tree.predict(x_test)
    confusion = confusion_matrix(y_test, predictions)
    acc = accuracy(confusion)

    return acc


# Example usage
if __name__ == '__main__':
    
    dataset_clean = np.array(np.loadtxt("../../wifi_db/clean_dataset.txt"))
    dataset_noisy = np.array(np.loadtxt("../../wifi_db/noisy_dataset.txt"))

    decision_tree_classifier = DecisionTreeClassifier()

    seed = 60012
    rg = default_rng(seed)

    print("Clean Dataset:")
    cross_validation(dataset_clean, decision_tree_classifier, num_folds=10, random_generator=rg)
    print()
    print("Noisy Dataset:")
    cross_validation(dataset_noisy, decision_tree_classifier, num_folds=10, random_generator=rg)
