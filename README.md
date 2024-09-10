# BERT-based-Multi-Label-Text-Classifier-with-Active-Learning-Continuous-Learning-and-Custom-AutoML
The repository uses a BERT-based multi-label text classifier for uncertain sample selection and real-time model improvement, with Optuna as a custom AutoML framework for hyperparameter optimization. This model is designed for large-scale classification tasks, especially review text, and can be fine-tuned based on user feedback.

## Features
* BERT-based Multi-Label Text Classification: Fine-tuned BERT models (with support for DistilBERT, RoBERTa) for multi-label classification tasks.
* Custom AutoML Framework (Optuna): Automates the tuning of critical hyperparameters (batch size, learning rate, sequence length, etc.) for improved performance.
* Active Learning: Dynamically queries uncertain samples from unlabeled data for labeling, enhancing model performance incrementally.
* Continuous Learning: Continuously monitors for new feedback data and fine-tunes the model in real-time.
* Efficient Data Preprocessing: Handles text cleaning, category encoding, price normalization, and other preprocessing tasks automatically.

## Dataset
The dataset used for this project is the Amazon Sales Dataset, which consists of various attributes like product reviews, categories, pricing information, and more.

## Data Overview
The dataset includes the following columns relevant to this model:

* review_title: Title of the product review.
* review_content: Full content of the product review.
* category: Multi-label categories for each review, separated by |.
* user_name: Username of the reviewer.
* discounted_price: Discounted price of the product.
* actual_price: Actual price of the product.
* product_id: Unique identifier for the product.

## Download and Use
* To download the dataset:https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset/data
* Place the amazon.csv file in the root directory of this project.

## Setup and Installation
### Requirements
Ensure the following packages are installed:
* Python 3.6+
* PyTorch
* Transformers (Hugging Face)
* Optuna (Custom AutoML)
* scikit-learn
* Pandas
* Matplotlib
* Seaborn
* tqdm

## Model Pipeline
1. Data Preprocessing
The preprocessing steps involve:

Cleaning text to remove special symbols.
Validating usernames for proper format.
Splitting and encoding categories using MultiLabelBinarizer.
Normalizing prices (discounted and actual).
Removing duplicate rows and irrelevant columns (img_link, product_link).
