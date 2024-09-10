import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import optuna
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import time



# Define Dataset class at the top level
class ReviewDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        review_text = str(self.data.iloc[index, 0])
        labels = torch.tensor(self.data.iloc[index, 1], dtype=torch.float)

        encoding = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

def calculate_accuracy(preds, labels, threshold=0.5):
    preds = preds.sigmoid()  # Apply sigmoid to get probabilities
    preds = (preds > threshold).float()  # Binarize predictions
    correct = (preds == labels).float()  # Compare with true labels
    accuracy = correct.sum() / labels.numel()  # Average over all elements
    return accuracy.item()

def preprocess_data(df):
    df['text'] = df['review_title'] + " " + df['review_content']
    return df[['text', 'category_encoded']]

def create_data_loader(data, tokenizer, max_len, batch_size):
    dataset = ReviewDataset(data, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)

def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    epoch_loss = 0
    for step, batch in enumerate(tqdm(data_loader, desc="Training"), 1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        preds = outputs.logits
        accuracy = calculate_accuracy(preds, labels)

        epoch_loss += loss.item()

        if step % 100 == 0 or step == len(data_loader):
            print(f"Step [{step}/{len(data_loader)}] - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    return epoch_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model = model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc="Evaluating"), 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            preds = outputs.logits
            accuracy = calculate_accuracy(preds, labels)

            epoch_loss += loss.item()

            if step % 100 == 0 or step == len(data_loader):
                print(f"Step [{step}/{len(data_loader)}] - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    return epoch_loss / len(data_loader)

def active_learning_query(model, tokenizer, device, unlabeled_data, mlb, max_len, threshold=0.5):
    """
    Select examples with uncertain predictions for labeling.
    """
    model.eval()
    uncertain_samples = []
    with torch.no_grad():
        for index, row in tqdm(unlabeled_data.iterrows(), total=unlabeled_data.shape[0], desc="Active Learning Query"):
            review_text = row['text']
            inputs = tokenizer.encode_plus(
                review_text,
                max_length=max_len,
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                truncation=True
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs.logits.sigmoid()

            # Calculate uncertainty based on proximity to threshold
            uncertainty = torch.abs(probs - threshold)
            min_uncertainty, _ = torch.min(uncertainty, dim=1)
            # Select samples where minimum uncertainty is below a small value (e.g., 0.1)
            if (min_uncertainty < 0.1).any():
                uncertain_samples.append(row)

    return pd.DataFrame(uncertain_samples)

def continuous_learning_with_active_learning(feedback_folder, tokenizer, model, device, unlabeled_data, mlb, max_len, batch_size, learning_rate=2e-5, check_interval=60):
    """
    Continuously monitor for new feedback data and fine-tune the model using that data.
    Implements active learning by querying uncertain samples.
    After 30 seconds of checking, it asks the user whether to exit or continue.
    """
    processed_files = set()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    exit_loop = False

    while not exit_loop:
        feedback_files = [f for f in os.listdir(feedback_folder) if f.endswith('.csv') and f not in processed_files]

        for feedback_file in feedback_files:
            feedback_path = os.path.join(feedback_folder, feedback_file)
            print(f"Processing feedback file: {feedback_file}")

            # Load feedback data and handle missing/invalid categories
            feedback_df = pd.read_csv(feedback_path)

            # Ensure category column is treated as strings and handle NaN values
            if 'category' not in feedback_df.columns:
                print(f"'category' column not found in {feedback_file}. Skipping file.")
                processed_files.add(feedback_file)
                continue

            feedback_df['category'] = feedback_df['category'].fillna('').astype(str)
            feedback_df['category'] = feedback_df['category'].apply(
                lambda x: x.split('|') if isinstance(x, str) else []
            )

            # Use the existing MultiLabelBinarizer to encode categories
            try:
                feedback_df['category_encoded'] = list(mlb.transform(feedback_df['category']))
            except ValueError as e:
                print(f"Error in encoding categories for {feedback_file}: {e}. Skipping file.")
                processed_files.add(feedback_file)
                continue

            # Combine title and content into 'text' for model input
            feedback_df['text'] = feedback_df['review_title'] + " " + feedback_df['review_content']
            feedback_data = feedback_df[['text', 'category_encoded']]

            # Create DataLoader for new feedback
            feedback_loader = create_data_loader(feedback_data, tokenizer, max_len, batch_size)

            # Fine-tune model on new feedback data
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(feedback_loader, desc=f"Fine-tuning on {feedback_file}"), 1):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(feedback_loader)
            print(f"Processed {feedback_file}: Fine-tuning Loss = {avg_loss:.4f}")

            # Mark the file as processed
            processed_files.add(feedback_file)

        # Perform active learning on the unlabeled data
        if not unlabeled_data.empty:
            print("Performing active learning query on unlabeled data...")
            uncertain_samples = active_learning_query(model, tokenizer, device, unlabeled_data, mlb, max_len, threshold=0.5)
            print(f"Uncertain samples to be labeled: {len(uncertain_samples)}")

        # Wait for the specified interval before checking again
        time.sleep(check_interval)

        # Ask user if they want to continue or exit after 30 seconds
        user_input = input(f"Do you want to continue waiting for feedback? (yes to continue, no to exit): ").strip().lower()
        if user_input != 'yes':
            print("Exiting continuous learning.")
            exit_loop = True
        else:
            print(f"Waiting for {check_interval} seconds before next check...")

def main():
    # Load the processed dataset
    df = pd.read_csv('amazon.csv')



    # Remove duplicates by keeping the first occurrence and drop rows with any null values
    df = df.drop_duplicates(subset='product_id').dropna()

    # Remove 'img_link' and 'product_link' columns if they exist
    df = df.drop(columns=['img_link', 'product_link'], errors='ignore')

    # Function to clean special symbols from text
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        return text

    # Function to validate usernames
    def is_valid_username(username):
        if isinstance(username, str):
            return bool(re.search(r'[A-Za-z0-9]', username))
        return False

    # Apply text cleaning
    df['review_title'] = df['review_title'].apply(clean_text)
    df['review_content'] = df['review_content'].apply(clean_text)
    df = df[df['user_name'].apply(is_valid_username)]

    # Split categories
    df['category'] = df['category'].astype(str)
    categories_split = df['category'].str.split('|', expand=True)
    df['first_category'] = categories_split[0]
    df['second_category'] = categories_split[1].fillna('')
    df['third_category'] = categories_split[2].fillna('')

    # Clean price columns
    df['discounted_price'] = df['discounted_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
    df['actual_price'] = df['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)

    # Save and reload processed data
    df.to_csv('input.csv', index=False)
    df = pd.read_csv('input.csv')



    # Initialize MultiLabelBinarizer and fit on all categories
    mlb = MultiLabelBinarizer()
    df['category'] = df['category'].apply(lambda x: x.split('|'))
    mlb.fit(df['category'])

    # Encode categories
    df['category_encoded'] = list(mlb.transform(df['category']))

    # Preprocess data
    df = preprocess_data(df)
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Load unlabeled data (assuming 'unlabeled.csv' exists)
    # If not available, create an empty DataFrame
    if os.path.exists('unlabeled.csv'):
        unlabeled_data = pd.read_csv('unlabeled.csv')
        unlabeled_data['text'] = unlabeled_data['review_title'] + " " + unlabeled_data['review_content']
        unlabeled_data = unlabeled_data[['text']]
        print(f"Loaded {len(unlabeled_data)} samples from 'unlabeled.csv'.")
    else:
        print("Unlabeled data file 'unlabeled.csv' not found. Proceeding without active learning.")
        unlabeled_data = pd.DataFrame(columns=['text'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define the Optuna objective function inside main to access 'train_data', 'test_data', 'mlb', etc.
    def objective(trial):
        # Suggest hyperparameters
        BATCH_SIZE = trial.suggest_categorical('batch_size', [16, 32, 64])
        MAX_LEN = trial.suggest_categorical('max_len', [128, 256, 512])
        LEARNING_RATE = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        EPOCHS = trial.suggest_int('epochs', 1, 2)
        # Suggest model architecture
        model_name = trial.suggest_categorical('model', ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base'])

        print(f"Trial {trial.number}: Model={model_name}, Batch Size={BATCH_SIZE}, Max Len={MAX_LEN}, LR={LEARNING_RATE}, Epochs={EPOCHS}")

        # Load the suggested model
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(mlb.classes_),
            problem_type="multi_label_classification"
        )
        model = model.to(device)

        # Create DataLoaders
        train_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
        test_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            train_loss = train_epoch(model, train_loader, optimizer, device)
            test_loss = eval_model(model, test_loader, device)
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Use test_loss as the objective to minimize
        return test_loss

    # Run Optuna study
    study = optuna.create_study(direction='minimize')
    print("Starting hyperparameter optimization with Optuna...")
    study.optimize(objective, n_trials=2)

    # Best hyperparameters
    best_params = study.best_params
    print('Best hyperparameters:', best_params)

    # Extract best parameters
    BATCH_SIZE = best_params['batch_size']
    MAX_LEN = best_params['max_len']
    LEARNING_RATE = best_params['lr']
    EPOCHS = best_params['epochs']
    model_name = best_params['model']

    # Initialize the final model with best hyperparameters
    print(f"Training final model with best parameters: Model={model_name}, Batch Size={BATCH_SIZE}, Max Len={MAX_LEN}, LR={LEARNING_RATE}, Epochs={EPOCHS}")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification"
    )
    model = model.to(device)

    # Create DataLoaders
    train_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
    test_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train the final model
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = eval_model(model, test_loader, device)
        print(f'Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Save model and tokenizer
    os.makedirs("model", exist_ok=True)
    model.save_pretrained("model/bert_multi_label_model")
    tokenizer.save_pretrained("model/bert_multi_label_tokenizer")
    print("Model and tokenizer saved to 'model/' directory.")

    # Start continuous learning with active learning
    feedback_folder = "feedback_data"  # Path to folder where feedback files are stored
    os.makedirs(feedback_folder, exist_ok=True)
    print(f"Starting continuous learning. Monitoring '{feedback_folder}' for new feedback files...")

    continuous_learning_with_active_learning(
        feedback_folder=feedback_folder,
        tokenizer=tokenizer,
        model=model,
        device=device,
        unlabeled_data=unlabeled_data,
        mlb=mlb,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        check_interval=15  # Check every 15 seconds
    )

if __name__ == "__main__":
    main()
