from src.models.train_model import DecisionTreeClassifier
import numpy as np



def confusion_matrix(y_true, y_pred, num_classes):

    """
    Calculate the confusion matrix for a classification problem.

    Parameters:
    y_true (np.ndarray): The true labels.
    y_pred (np.ndarray): The predicted labels.
    num_classes (int): The number of classes.

    Returns:
    np.ndarray: The confusion matrix (num_classes x num_classes).
   
    """

    cm = np.zeros((num_classes, num_classes), dtype = int)

    for t, p in zip(y_true, y_pred):
        cm[int(t) - 1, int(p) - 1] += 1
    
    return cm

def calculate_metrics(conf_matrix):

    """
    Calculate the accuracy, precision, recall, and F1 score from a confusion matrix.

    Parameters:
    conf_matrix (np.ndarray): The confusion matrix.

    Returns:
    float: The accuracy.
    np.ndarray: The precision for each class.
    np.ndarray: The recall for each class.  
    np.ndarray: The F1 score for each class.

    """

    num_classes = conf_matrix.shape[0]

    true_positives = np.diag(conf_matrix)
    false_positives = np.sum(conf_matrix, axis=0) - true_positives
    false_negatives = np.sum(conf_matrix, axis=1) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)

    accuracy = np.sum(true_positives) / np.sum(conf_matrix)

    return accuracy, precision, recall, f1_score


def cross_validation_evaluate(dataset, model, num_classes, num_folds=10):

    """
    Perform k-fold cross-validation to evaluate a model.

    Parameters:
    dataset (np.ndarray): The dataset to evaluate.
    model (object): The model to evaluate.
    num_classes (int): The number of classes.
    num_folds (int): The number of folds.

    Returns:
    np.ndarray: The confusion matrix.
    float: The accuracy.
    np.ndarray: The precision for each class.
    np.ndarray: The recall for each class.
    np.ndarray: The F1 score for each class.

    """

    
    np.random.shuffle(dataset)
    fold_size = len(dataset) // num_folds
    cumulative_conf_matrix = np.zeros((num_classes, num_classes), dtype = int)

    for i in range(num_folds):
        # Split the dataset into train and test sets for this fold
        start, end = i * fold_size, (i + 1) * fold_size
        test_data = dataset[start:end]
        train_data = np.concatenate([dataset[:start], dataset[end:]], axis=0)

        # Train the model
        model.fit(train_data)

        # Separate the features and labels for test data
        x_test, y_test = test_data[:, :-1], test_data[:, -1]

        # Predict the labels
        predictions = model.predict(x_test)

        # Get confusion matrix for this fold and accumulate 
        fold_conf_matrix = confusion_matrix(y_test, predictions, num_classes)
        cumulative_conf_matrix += fold_conf_matrix

    # Calculate metrics
    accuracy, precision, recall, f1_score = calculate_metrics(cumulative_conf_matrix)
    return cumulative_conf_matrix, accuracy, precision, recall, f1_score
    

def evaluate(test_data, model, num_classes):

    cm, acc, prec, rec, f1= cross_validation_evaluate(test_data, model, num_classes, num_folds=10)
    print("Confusion Matrix: ", cm)
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", rec)
    print("F1 Score: ", f1)

    return acc

# Example usage

if __name__ == "__main__":
    
    dataset_clean = np.loadtxt("../../wifi_db/clean_dataset.txt")
    dataset_noisy = np.loadtxt("../../wifi_db/noisy_dataset.txt")

    num_classes = 4

    decision_tree_classifier = DecisionTreeClassifier()

    print("Clean Dataset:")
    evaluate(dataset_clean, decision_tree_classifier, num_classes)
    print("\nNoisy Dataset:")
    evaluate(dataset_noisy, decision_tree_classifier, num_classes)


    