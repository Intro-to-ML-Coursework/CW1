# Decision Tree Training, Visualisation, and Evaluation

This project involves training a decision tree for predicting the room number using Wi-Fi signal data, visualizing the
trained decision tree, and evaluating its performance. The code is structured into several modules to handle training,
visualisation, and evaluation tasks.

## Project Structure
- src
  - models
    - train_model.py
  - utils
    - utils.py
  - visualisation
    - tree.png
    - tree_top.png
    - visualise.py
- wifi_db
  - clean_dataset.txt
  - noisy_dataset.txt
- .gitignore
- README.md
- requirements.txt

## Getting Started

### 1. Set up the virtual environment
Ensure you have Python 3.12 installed.\
Once Python 3.12 is installed, follow the steps below to set up the project:

```bash
# Navigate to the project directory
cd CW1

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### 2. Train the Decision Tree
To train the decision tree using the provided dataset, run the ```train_model.py``` script. This will read the data from
the ```wifi_db/clean_dataset.txt``` file, train a decision tree, and print the tree (in the form of a dictionary) to the
console.

```bash
# Navigate to the models directory
cd src/models

# Run the training script
python train_model.py
```

### 3. Visualise the Trained Decision Tree
To visualise the trained decision tree, run the ```visualise.py``` script. This will generate a plot for the decision
tree and save it as ```tree.png``` in the ```visualisation/``` directory.

```bash
# Navigate to the visualisation directory
cd ../visualisation

# Run the visualisation script
python visualise.py
```

### 4. Evaluate the Decision Tree
The ```evaluate.py``` script will train the decision tree using the provided data and evaluate its performance by
calculating and printing the following metrics:
- Confusion matrix
- Accuracy
- Recall and precision rates per class
- F1-measures derived from the recall and precision rates of the previous step

To run the evaluation:
```bash
# Navigate to the evaluation directory
cd ../evaluation

# Run the evaluation script
python evaluate.py
```

## Dataset
The Wi-Fi signal data is provided in the ```wifi_db/``` directory. The files included are:
- clean_dataset.txt: The clean dataset used for training.
- noisy_dataset.txt: A noisy dataset for testing and evaluation.

## Dependencies
The project dependencies are listed in ```requirements.txt```. They include:
- matplotlib==3.9.2
- numpy==2.1.1
- scipy==1.14.1

Make sure you have these installed by running ```pip install -r requirements.txt```.
