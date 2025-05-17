# ====================================== IMPORTS ======================================

# Library torch imported to build and train a GPT model for sentiment analysis
# torch imports the PyTorch library for tensor operations and GPU acceleration
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
# Library wandb imported to log and visualize training metrics
import wandb
# Library pandas imported to handle data manipulation and analysis
import pandas as pd
# Library nltk imported for natural language processing tasks
from nltk.corpus import stopwords
import nltk
# Library os imported to handle file and directory operations
import os
# For the tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, TensorDataset
from nltk.stem import WordNetLemmatizer

print("Imports done")

# ====================================== GPU DETECTION ======================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(100)

print("GPU detection done")

# ====================================== HYPERPARAMETERS ======================================
vocab_n = 10000
sequence_length = 100
batch_size = 32
num_workers = 0

learning_rate = 1e-3
num_epochs = 200

input_dim = 128
hidden_dim1 = 128
hidden_dim2 = 256
hidden_dim3 = 512
linear_layer1 = 256
linear_layer2 = 128
layer_dim = 1
output_dim = 1
number_of_lstm_layers = 3
number_of_linear_models = 3


eval_every = 1
save_every = 10

dataset_part = 1
dropout = 0.2

print("Hyperparameters done")

# ====================================== CHECKPOINTS ======================================

model_name = "LSTMModelBCEWithLogits"
directory = "../vilour/LSTM"
checkpoint = f"{directory}/{model_name}.pt"

print("Checkpoints done")

# ====================================== WAND B ======================================
wandb.init(
    project="LSTM",
    config={
        "model": model_name,
        "batch_size": batch_size,
        "vocab_n": vocab_n,
        "sequence_legth": sequence_length,

        "input_dim": input_dim,
        "hidden_dim1": hidden_dim1,
        "hidden_dim2": hidden_dim2,
        "hidden_dim3": hidden_dim3,
        "linear_layer1": linear_layer1,
        "linear_layer2": linear_layer2,
        "layer_dim": layer_dim,
        "output_dim": output_dim,
        "number_of_lstm_layers": number_of_lstm_layers,
        "number_of_linear_models": number_of_linear_models,

        "num_workers": num_workers,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,


        "eval_every": eval_every,
        "part of the dataset": dataset_part,
        "dropout": dropout,
    }
)

print("WandB done")

# ====================================== DATA ======================================

# Dataset loading
file_path = "rotten_tomatoes_movie_reviews.csv"
dataSet = pd.read_csv(file_path)

print("Dataset loaded")

# Decreasing the size of the dataset to speed up the training process
if dataset_part != 1:
    dataSet = dataSet.sample(frac=dataset_part, random_state=42)

print("Dataset size decreased")

# DATASET PREPROCESSING

# 1. Replace POSITIVE and NEGATIVE with 1 and 0
dataSet['scoreSentiment'] = dataSet['scoreSentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})
# 2. Removes punctuation and special characters
dataSet['reviewText'] = dataSet['reviewText'].str.replace('[^\w\s]', '', regex=True)

# 3. Removes stop words which do not add value to the analysis
# Stopwords download
# nltk.download("stopwords")
# stop_words = set(stopwords.words('english'))

# dataSet['reviewText'] = dataSet['reviewText'].apply(
#     lambda x: ' '.join([word for word in x.split() if word not in stop_words])
# )
# not or never being stop words and having a strong impact on the sentiment,
# stop words should not been removed

# 4. Lemmatization to reduce words to their base form
# nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()
# dataSet['reviewText'] = dataSet['reviewText'].apply(
#     lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
# )

# 5. Convert to lowercase
dataSet['reviewText'] = dataSet['reviewText'].str.lower()

# 6. Remove extra spaces
dataSet['reviewText'] = dataSet['reviewText'].str.replace(r'\s+', ' ', regex=True).str.strip()

# 7. Remove empty reviews
dataSet = dataSet[dataSet['reviewText'].str.strip() != '']

print("Dataset preprocessed")

# Preprocessing ideas 2 and 3 comes from the following link:
# https://medium.com/@felixmutai/tokenization-in-sentiment-analysis-cd386a08fab

# Dataset conversion to lists in order to be used by the tokenizer 
# Initially, datasets are in pandas DataFrame format and need to be converted to lists for tokenization
reviews = dataSet['reviewText'].tolist()
labels = dataSet['scoreSentiment'].tolist()

# TOKENIZATION 

# Convert the reviews and labels to numpy arrays for easier manipulation
X = dataSet["reviewText"].fillna("").astype(str).to_numpy()
y = dataSet["scoreSentiment"].to_numpy()
print(X[:10], "and y = ", y[:10])
print(X.dtype)

# Initialize the tokenizer and fit it on the reviews
tokenizer = Tokenizer(num_words=vocab_n, oov_token="<UNK>")  # <UNK> is the out-of-vocabulary token
tokenizer.fit_on_texts(X)

# Convert the reviews to sequences of integers
X_sequences = tokenizer.texts_to_sequences(X)

# Define the padding token ID
X_padded = pad_sequences(X_sequences, padding='post', maxlen=sequence_length)

# Convert the padded sequences to PyTorch tensors
encoded_reviews = torch.tensor(X_padded, dtype=torch.long).to(device)
encoded_labels = torch.tensor(y, dtype=torch.long).to(device)

print("Tokenization and padding done")

n = int(0.8 * len(encoded_reviews))

# Define the maximum length for padding
train_reviews = encoded_reviews[:n]
test_reviews = encoded_reviews[n:]

train_labels = encoded_labels[:n]
test_labels = encoded_labels[n:]

train_dataset = TensorDataset(train_reviews, train_labels)
validation_dataset = TensorDataset(test_reviews, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

print("Train and test sets split")


@torch.no_grad()
def evaluate(model, train_loader, validation_loader, criterion):
    model.eval()  # Mettre le modèle en mode évaluation
    results = {}

    for split, loader in [("train", train_loader), ("validation", validation_loader)]:
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)  # Déplacer les données sur le bon périphérique
            logits = model(x)  # Obtenir les prédictions du modèle
            loss = criterion(logits, y.unsqueeze(-1).float())  # Calculer la perte
            total_loss += loss.item()

            # Convertir les logits en prédictions de classe
            predictions = torch.argmax(logits, dim=1)

            total_correct += (predictions == y).sum().item()
            total_samples += y.size(0)

        # Calculer la perte moyenne et l'accuracy pour ce split
        results[f"{split}_loss"] = total_loss / len(loader)
        results[f"{split}_accuracy"] = total_correct / total_samples

    model.train()  # Remettre le modèle en mode entraînement
    return results


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.embedding = nn.Embedding(vocab_n, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim2, layer_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim3, layer_dim, batch_first=True)

        self.linear1 = nn.Linear(hidden_dim3, linear_layer1)
        self.linear2 = nn.Linear(linear_layer1, linear_layer2)
        self.fc = nn.Linear(linear_layer2, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        out, (hn, cn) = self.lstm2(out)
        out, (hn, cn) = self.lstm3(out)
        out = out[:, -1, :]
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = self.dropout(out)
        return out

network = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim1, layer_dim=layer_dim, output_dim=output_dim)

device = torch.device(device)
network.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

print(f"Parameters: {sum(p.numel() for p in network.parameters())/1e6} M parameters")
def train(num_epochs, learning_rate):
    train_avg_loss = []
    test_avg_loss = []
    test_accuracy = []

    for i in range(num_epochs + 1):
        if i % 10 == 0:
            learning_rate /= 2
            optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        print(f"Epoch number : {i+1}")
        if i % eval_every == 0:
            # Evaluation on the test set
            results = evaluate(network, train_loader, validation_loader, criterion)
            print(f"Step {i}: Train Loss: {results['train_loss']:.4f}, Train Accuracy: {results['train_accuracy']:.4f}, "
              f"Validation Loss: {results['validation_loss']:.4f}, Validation Accuracy: {results['validation_accuracy']:.4f}")

            wandb.log({
                "step": i,
                "train_loss": results['train_loss'],
                "train_accuracy": results['train_accuracy'],
                "validation_loss": results['validation_loss'],
                "validation_accuracy": results['validation_accuracy']
            })

        train_losses = []
        test_losses = []

        network.train()

        for x, y in train_loader:
            pred = network(x)
            loss = criterion(pred, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % save_every == 0:
            torch.save(network.state_dict(), checkpoint)


    return train_avg_loss, test_avg_loss, test_accuracy


train_avg_loss, test_avg_loss, test_accuracy = train(num_epochs, learning_rate)
print("Training done")


