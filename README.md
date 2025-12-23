# Iris Classifier (Decision Tree)

## Overview
This project is an end-to-end machine learning example developed as part of an AI Development training programme.  
It demonstrates the complete supervised learning workflow using the classic Iris dataset and a Decision Tree classifier implemented with scikit-learn.
The project includes data loading, model training, prediction, evaluation, and visualization of results.

## Dataset
The Iris dataset contains 150 samples of iris flowers with four numerical features:
- Sepal length
- Sepal width
- Petal length
- Petal width

Each sample belongs to one of three classes:
- Setosa
- Versicolor
- Virginica

## Model
A **Decision Tree Classifier** is used in this project because it is simple, interpretable, and well-suited for small datasets.
The dataset is split into training (80%) and testing (20%) subsets to evaluate the model's performance on unseen data.

## Project Structure
```
iris-classifier/
├── notebooks/
│   └── iris_model.ipynb          # Step-by-step exploratory notebook
├── src/
│   └── train.py                  # Reproducible training script
├── outputs/
│   └── confusion_matrix.png      # Generated confusion matrix
├── requirements.txt
├── README.md
└── venv/
```

## Quick Start

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the training script
```bash
python src/train.py --test-size 0.2 --random-state 42
```

The script prints the model accuracy and saves a confusion matrix image to the `outputs/` directory.

## Results
The trained Decision Tree classifier achieves high accuracy on the test set.
Most classification errors occur between the Versicolor and Virginica classes, which is expected due to their similar characteristics.

## License
This project is licensed under the MIT License.