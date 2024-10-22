from random import random

import numpy as np
import matplotlib.pyplot as plt
from src.models.train_model import decision_tree_learning


def plot_decision_tree(tree, x_min=0., x_max=1., y=0.):
    """
    Recursively plots the decision tree using matplotlib.

    Parameters:
    tree (dict): The decision tree to visualize.
    x_min (float): The left boundary of the node's x-position.
    x_max (float): The right boundary of the node's x-position.
    y (float): The y-position of the node (corresponds to depth).

    Returns:
    None
    """
    if tree is None:
        return

    # Calculate the x-position of the current node
    x = (x_min + x_max) / 2

    # Generate a random color
    linecolor = (random(), random(), random())

    # Plot the current node
    if tree['attribute'] is None:
        # Leaf node
        plt.text(x, -y, f"Leaf: {tree['value']}", ha='center', va='center',
                 bbox=dict(facecolor='lightgreen', edgecolor='black'), fontsize=30)
    else:
        # Internal node
        plt.text(x, -y, f"X{tree['attribute']} < {tree['value']}", ha='center', va='center',
                 bbox=dict(facecolor='lightblue', edgecolor='black'), fontsize=30)

    # Plot the children recursively
    if tree['left'] is not None:
        # Draw a line between parent and left child
        child_x = (x_min + x) / 2
        plt.plot([x, child_x], [-y, -(y + 6)], color=linecolor)

        plot_decision_tree(tree['left'], x_min, x, y + 6)

    if tree['right'] is not None:
        # Draw a line between parent and right child
        child_x = (x_max + x) / 2
        plt.plot([x, child_x], [-y, -(y + 6)], color=linecolor)

        plot_decision_tree(tree['right'], x, x_max, y + 6)


def plot_and_save_to(tree, depth, path):
    """
    Plot the decision tree and save it to the given path.

    Parameters:
    tree (dict): The decision tree to visualize.
    depth (int): The depth of the tree.
    path (str): The path to save the plot to.

    Returns:
    None
    """
    # Set up the plot
    plt.figure(figsize=(60, 6 * depth))
    plt.axis('off')

    # Plot the decision tree
    plot_decision_tree(tree, depth)

    # Save the plot
    plt.savefig(path)


# Example usage
if __name__ == '__main__':
    dataset_clean = np.loadtxt("../../wifi_db/clean_dataset.txt")
    decisionTree, tree_depth = decision_tree_learning(dataset_clean, 0)

    plot_and_save_to(decisionTree, tree_depth, "tree.png")
