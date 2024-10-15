import numpy as np

dataset = np.loadtxt("../../wifi_db/clean_dataset.txt")
print(dataset.shape)

x = dataset[:, :-1]
print(x)
print(x.shape)

y = dataset[:, -1]
print(y)
print(y.shape)
