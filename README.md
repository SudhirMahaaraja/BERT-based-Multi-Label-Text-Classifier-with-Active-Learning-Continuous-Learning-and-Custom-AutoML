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
* To download the dataset: https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset/data
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

* Cleaning text to remove special symbols.
* Validating usernames for proper format.
* Splitting and encoding categories using MultiLabelBinarizer.
* Normalizing prices (discounted and actual).
* Removing duplicate rows and irrelevant columns (img_link, product_link).

2. Hyperparameter Tuning (Custom AutoML Framework with Optuna)
Optuna is used as the custom-made AutoML framework to optimize hyperparameters, including:

* Batch size (16, 32, 64)
* Maximum sequence length (128, 256, 512)
* Learning rate (log-uniform range between 1e-5 to 1e-3)
* Number of epochs (1, 2)
* Model architecture (bert-base-uncased, distilbert-base-uncased, roberta-base)

3. Training the Best Model
The best hyperparameters are extracted from Optuna, and the final model is trained. Both the training and testing processes compute loss and accuracy at each epoch.

4. Active Learning for Unlabeled Data
The model dynamically selects uncertain samples from unlabeled data, based on prediction probabilities. The selected samples are those with the highest uncertainty, measured by how close the prediction probabilities are to a defined threshold (0.5 by default).

5. Continuous Learning with Feedback Integration
The model continuously monitors a specified folder (feedback_data/) for new feedback data. Whenever new labeled data is added, the model is fine-tuned on it. This process can run indefinitely with regular checks every 15 seconds.

6. Model and Tokenizer Saving
The final model and tokenizer are saved to the model/ directory for later use.

## Customization Options
* Active Learning: You can modify the uncertainty threshold in active_learning_query() to select different levels of uncertainty for sample labeling.
* Feedback Interval: Adjust the check_interval parameter in the continuous_learning_with_active_learning() function to control how often the system checks for new feedback files.
* Optuna Trials: Change the n_trials argument in study.optimize() to control the number of AutoML iterations.

## Results and Output
* Model Files: The trained model and tokenizer are saved in the model/ directory.
* Logs: Training loss, evaluation loss, and active learning query results are printed to the console.
* Feedback: The model improves dynamically with new feedback files.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as needed.



