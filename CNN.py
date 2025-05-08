# ====================================== IMPORTS ======================================

# Library torch imported to build and train a GPT model for sentiment analysis
# torch imports the PyTorch library for tensor operations and GPU acceleration
import torch
import torch.nn as nn
from torch.nn import functional as F
# Library wandb imported to log and visualize training metrics
import wandb
# Library pandas imported to handle data manipulation and analysis
import pandas as pd
# Library tokenizers imported to handle tokenization of text data
from tokenizers import ByteLevelBPETokenizer
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
num_epochs = 50

input_features = sequence_length
output_features = 1

output_Features1 = 64
output_Features2 = 256
output_Features3 = 512
kernel_size = 3

output_Features_final = 512

output_linear1 = 256
output_linear2 = 128
output_linear3 = 64

number_of_conv_layers = 3
number_of_linear_layers = 3

eval_every = 1
save_every = 10

dataset_part = 0.10

print("Hyperparameters done")

# ====================================== CHECKPOINTS ======================================

model_name = "CNN_2Layer_20"
directory = "../vilour/CNN"
checkpoint = f"{directory}/{model_name}.pt"

print("Checkpoints done")

# ====================================== WAND B ======================================
wandb.init(
        project="CNN",
        config={
            "model": model_name,
            "batch_size": batch_size,
            "vocab_n": vocab_n,
            "sequence_legth": sequence_length,
            "num_workers": num_workers,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,

            "number of conv layers": number_of_conv_layers,
            "kernel size": kernel_size,
            "number of linear layers": number_of_linear_layers,

            "eval_every": eval_every,
            "part of the dataset": dataset_part,
            }
        )

print("WandB done")

# ====================================== DATA ======================================

# Dataset loading
file_path = "rotten_tomatoes_movies_reviews.csv"
dataSet = pd.read_csv(file_path)

print("Dataset loaded")

# Decreasing the size of the dataset to speed up the training process
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
X = dataSet["reviewText"].to_numpy()
y = dataSet["scoreSentiment"].to_numpy()

# Initialize the tokenizer and fit it on the reviews
tokenizer = Tokenizer(num_words=vocab_n, oov_token="<UNK>")  # <UNK> is the out-of-vocabulary token
tokenizer.fit_on_texts(X)

# Convert the reviews to sequences of integers
X_sequences = tokenizer.texts_to_sequences(X)

# Define the padding token ID
X_padded = pad_sequences(X_sequences, padding='post', maxlen=max_seq_len)

# Convert the padded sequences to PyTorch tensors
encoded_reviews = torch.tensor(X_padded, dtype=torch.long).to(device)
encoded_labels = torch.tensor(y, dtype=torch.long).to(device)

print("Tokenization and padding done")

# Padding function
def pad_sequences(sequences, max_length, pad_token_id):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            # Padding if the sequence is too short
            # dtype=torch.long is used to ensure that the padding token is of the same type as the input sequences
            seq = torch.cat([seq, torch.full((max_length - len(seq),), pad_token_id, device=device, dtype=torch.long)])
        else:
            # Truncating if the sequence is too long
            seq = seq[:max_length]
        padded_sequences.append(seq)
    return torch.stack(padded_sequences)

# TRAIN AND TEST SETS SPLIT 

n = int(0.8 * len(encoded_reviews))

# Define the maximum length for padding
# Limit the maximum length of the sequences to a fixed size (block_size) for training
max_length = min(block_size, max(len(seq) for seq in encoded_reviews))

train_reviews = encoded_reviews[:n]
test_reviews = encoded_reviews[n:]

train_labels = encoded_labels[:n]
test_labels = encoded_labels[n:]

train_dataset = TensorDataset(train_reviews, train_labels)
validation_dataset = TensorDataset(test_reviews, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

print("Train and test sets split")

# DATA LOADER

# Generate mini-batch of data for training
def get_batch(data, labels):
    # Randomly select batch_size sequences from the dataset
    # In example of teacher, len(data) - block_size is used because 
    # the model needs the next token to check if the prediction is
    # correct or not. In our case, we do not need the next token.
    ix = torch.randint(len(data), (batch_size,))
    # Select the corresponding labels for the selected sequences
    x = torch.stack([data[i] for i in ix])
    # Select the corresponding labels for the selected sequences
    y = torch.stack([labels[i] for i in ix])
    # Move the data and labels to the specified device
    x, y = x.to(device), y.to(device)
    # Return the mini-batch of data and labels
    return x, y



for i in range(number_of_conv_layers):
    input_features = (input_features - 2) // 2


class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Embedding(vocab_n, 15),
            nn.Conv1d(15, output_Features1, 3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(output_Features1, output_Features2, 3),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(output_Features2, output_Features3, 3),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(input_features * output_Features_final, output_linear1),
            nn.ReLU(),
            nn.Linear(output_linear1, output_linear2),
            nn.ReLU(),
            nn.Linear(output_linear2, output_linear3),
            nn.ReLU(),
            nn.Linear(output_linear3, output_features),
            nn.Sigmoif()
        )

network = CNN()

device = torch.device(device)
network.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

print(f"Parameters: {sum(p.numel() for p in network.parameters())/1e6} M parameters")
def train(num_epochs):
    train_avg_loss = []
    test_avg_loss = []
    test_accuracy = []

    for i in range(num_epochs + 1):
        if i % eval_every == 0:
            # Evaluation on the test set
            results = evaluate(lm, train_loader, validation_loader, criterion)
            print(f"Step {i}: Train Loss: {results['train_loss']:.4f}, Train Accuracy: {results['train_accuracy']:.4f}, "
              f"Validation Loss: {results['validation_loss']:.4f}, Validation Accuracy: {results['validation_accuracy']:.4f}")

            wandb.log({
                "step": i,
                "train_loss": results['train_loss'],
                "train_accuracy": results['train_accuracy'],
                "validation_loss": results['validation_loss'],
                "validation_accuracy": results['validation_accuracy']
            })

        print(f"Epoch number : {i+1}")
        train_losses = []
        test_losses = []

        network.train()

        x, y = get_batch(train_reviews, train_labels)
        pred = network(x)
        loss = criterion(pred, y)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % save_every == 0:
            torch.save(network.state_dict(), checkpoint)


    return train_avg_loss, test_avg_loss, test_accuracy


train_avg_loss, test_avg_loss, test_accuracy = train(num_epochs)
print("Training done")

