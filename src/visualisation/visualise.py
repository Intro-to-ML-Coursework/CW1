import numpy as np
import matplotlib.pyplot as plt
from src.models.train_model import decision_tree_learning

def plot_decision_tree(tree, depth, current_depth=0, x_min=0, x_max=1, y=0):
    """
    Recursively plots the decision tree using matplotlib.

    Parameters:
    tree (dict): The decision tree to visualize.
    depth (int): The depth of the decision tree.
    current_depth (int): The current depth of the node (used for recursion).
    x_min (float): The left boundary of the node's x-position.
    x_max (float): The right boundary of the node's x-position.
    y (float): The y-position of the node (corresponds to depth).
    """
    if tree is None:
        return

    # Calculate the x-position of the current node
    x = (x_min + x_max) / 2

    # Plot the current node
    if tree['attribute'] is None:
        # Leaf node
        plt.text(x, -y, f"Leaf: {tree['value']}", ha='center', va='center',
                 bbox=dict(facecolor='lightblue', edgecolor='black'))
    else:
        # Internal node
        plt.text(x, -y, f"Attr: {tree['attribute']}\nVal: {tree['value']}", ha='center', va='center',
                 bbox=dict(facecolor='lightgreen', edgecolor='black'))

    # Plot the children recursively
    if tree['left'] is not None:
        # Draw a line to the left child
        child_x = (x_min + x) / 2
        plt.plot([x, child_x], [-y, -(y + 1)], 'k-')  # Draw a line between parent and child
        plot_decision_tree(tree['left'], depth, current_depth + 1, x_min, x, y + 1)

    if tree['right'] is not None:
        # Draw a line to the right child
        child_x = (x_max + x) / 2
        plt.plot([x, child_x], [-y, -(y + 1)], 'k-')  # Draw a line between parent and child
        plot_decision_tree(tree['right'], depth, current_depth + 1, x, x_max, y + 1)


# Example usage
if __name__ == '__main__':
    dataset_clean = np.loadtxt("../../wifi_db/clean_dataset.txt")
    tree, tree_depth = decision_tree_learning(dataset_clean, 0)

    # Set up the plot
    plt.figure(figsize=(10, tree_depth))  # Dynamically adjust the figure size based on depth
    plt.axis('off')  # Turn off the axis

    # Plot the decision tree
    plot_decision_tree(tree, tree_depth)

    # Display the plot
    plt.show()


