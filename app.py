import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse
import numpy as np
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Define the category ID to name mapping
id_to_category = {
    0: 'Constructive Criticism',
    1: 'Emotional',
    2: 'Hate/Abuse',
    3: 'Irrelevant/Spam',
    4: 'Praise/Support',
    5: 'Question/Suggestion',
    6: 'Threat'
}

category_to_id = {v: k for k, v in id_to_category.items()}

# Load the pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
# Re-adding low_cpu_mem_usage=True to prevent memory errors.
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(id_to_category), low_cpu_mem_usage=True)

# Flag to check if the fine-tuned model is loaded
is_tuned_model_loaded = False

# Load the fine-tuned model weights (assuming saved in ./results/checkpoint-XYZ)
try:
    # Find the latest checkpoint directory
    checkpoints = [d for d in os.listdir('./results') if os.path.isdir(os.path.join('./results', d)) and 'checkpoint' in d]
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
    model_path = os.path.join('./results', latest_checkpoint)
    model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
    is_tuned_model_loaded = True
    print(f"Loaded fine-tuned model from {model_path}")
except Exception as e:
    print(f"Could not load fine-tuned model weights: {e}")
    print("Using the base pre-trained model instead.")

def predict_comment_category(comment):
    """Predicts the category of a single comment string."""
    # Tokenize and encode the input comment
    inputs = tokenizer(comment, return_tensors='pt', truncation=True, padding=True, max_length=128)

    # Move tensors and model to the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Explicitly move the model to the device to materialize the meta tensors
    model.to(device)

    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Make a prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class ID
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    # Map the ID back to the category name
    predicted_category = id_to_category[predicted_class_id]

    return predicted_category

# Custom Dataset class for training
class CommentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Ensure labels are of type torch.long for the loss function
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def train_model(df):
    """Trains the model on a given DataFrame."""
    st.info("Starting model training...")

    # Pre-process the data
    df['label'] = df['label'].map(category_to_id)
    
    # Check for and drop rows with invalid labels
    invalid_labels_count = df['label'].isnull().sum()
    if invalid_labels_count > 0:
        st.warning(f"Warning: Dropped {invalid_labels_count} comments with invalid labels. Please ensure all labels match the defined categories.")
        df = df.dropna(subset=['label'])
    
    comments = df['comment'].tolist()
    labels = df['label'].tolist()

    # Tokenize comments
    encodings = tokenizer(comments, truncation=True, padding=True, max_length=128)
    dataset = CommentDataset(encodings, labels)

    # Configure training
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(os.path.join('./results', 'final_model'))
    st.success("Model training complete and saved successfully! Please refresh the application to load the new model.")

# Streamlit App Layout
st.title("Comment Category Predictor")
st.write("This application predicts the category of comments using a fine-tuned DistilBERT model.")

# ---
# Streamlit UI
# ---

# Display a warning if the fine-tuned model was not loaded
if not is_tuned_model_loaded:
    st.warning("No fine-tuned model found. Predictions may be inaccurate. Please train a new model below.")

# Text input for single comment prediction
st.header("Predict a Single Comment")
single_comment_input = st.text_area("Enter a comment:", "")

if st.button("Predict Single Comment"):
    if single_comment_input:
        predicted_category = predict_comment_category(single_comment_input)
        st.write(f"**Predicted Category:** {predicted_category}")
    else:
        st.warning("Please enter a comment to predict.")

# File uploader for batch prediction
st.header("Predict Categories from a File")
uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])

if uploaded_file is not None:
    try:
        if uploaded_file.type == "text/csv":
            df_comments = pd.read_csv(uploaded_file)
        elif uploaded_file.type == "application/json":
            df_comments = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or JSON file.")
            st.stop()

        if 'comment' not in df_comments.columns:
            st.error("Error: The file must contain a column named 'comment'.")
        else:
            st.write("File uploaded successfully. Predicting categories...")
            df_comments['predicted_category'] = df_comments['comment'].apply(predict_comment_category)

            st.subheader("Predictions")
            st.dataframe(df_comments)

            # Display category distribution plot
            st.subheader("Category Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df_comments, y='predicted_category', ax=ax, order=df_comments['predicted_category'].value_counts().index)
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---
# New Section for Model Training
# ---

st.header("Train a New Model")
st.write("Upload a CSV file with 'comment' and 'label' columns to fine-tune the model for your data.")

training_file = st.file_uploader("Upload a CSV file for training", type=["csv"])

if st.button("Train Model"):
    if training_file is not None:
        try:
            df_train = pd.read_csv(training_file)
            if 'comment' not in df_train.columns or 'label' not in df_train.columns:
                st.error("Error: The training CSV file must contain columns named 'comment' and 'label'.")
            else:
                train_model(df_train)
        except Exception as e:
            st.error(f"Error during training: {e}")
    else:
        st.warning("Please upload a CSV file to begin training.")
