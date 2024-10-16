import numpy as np
import matplotlib.pyplot as plt
from src.models.train_model import decision_tree_learning

dataset_clean = np.loadtxt("../../wifi_db/clean_dataset.txt")
print(decision_tree_learning(dataset_clean, 0))
