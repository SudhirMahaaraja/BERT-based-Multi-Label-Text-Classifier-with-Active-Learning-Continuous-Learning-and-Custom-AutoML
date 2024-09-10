# BERT-based-Multi-Label-Text-Classifier-with-Active-Learning-Continuous-Learning-and-Custom-AutoML
The repository uses a BERT-based multi-label text classifier for uncertain sample selection and real-time model improvement, with Optuna as a custom AutoML framework for hyperparameter optimization. This model is designed for large-scale classification tasks, especially review text, and can be fine-tuned based on user feedback.

## Features
### BERT-based Multi-Label Text Classification: Fine-tuned BERT models (with support for DistilBERT, RoBERTa) for multi-label classification tasks.
### Custom AutoML Framework (Optuna): Automates the tuning of critical hyperparameters (batch size, learning rate, sequence length, etc.) for improved performance.
### Active Learning: Dynamically queries uncertain samples from unlabeled data for labeling, enhancing model performance incrementally.
### Continuous Learning: Continuously monitors for new feedback data and fine-tunes the model in real-time.
### Efficient Data Preprocessing: Handles text cleaning, category encoding, price normalization, and other preprocessing tasks automatically.
