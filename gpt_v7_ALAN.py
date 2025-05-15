# Remarks : 
# 1. GPT-style models are made for generation, not classification
# 2. GPT 2/3/4 better for classification then 7

# Search about scheduling in chap 4 of deepLearning
# Search about warm up
# Mixed Precision Training
# weightght Decay
# Data augmentation
# Adjust dynamically batch size

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
# from tokenizers import ByteLevelBPETokenizer
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

print("Imports done")

# ====================================== GPU DETECTION ======================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(100)

print("GPU detection done")

# ====================================== HYPERPARAMETERS ======================================

batch_size = 32
block_size = 512
n_embedding = 512
# Having a minimum large nb of blocks and heads is important to capture
# complex patterns in the data
n_blocks = 12
n_head = 8
dropout = 0.2
max_iters = 50000
learning_rate = 1e-4
eval_every = 500
eval_iters = 200
save_every = 1000
num_words = 50000
max_seq_len = 128
patience = 5

print("Hyperparameters done")

# ====================================== CHECKPOINTS ======================================

model_name = "sentiment-analysis-gpt"
directory = "./checkpoints"
os.makedirs(directory, exist_ok=True)
checkpoint = f"{directory}/{model_name}.pt"

print("Checkpoints done")

# ====================================== WAND B ======================================

wandb.init(
    project="sentiment-analysis-gpt",
    config={
        "model": model_name,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embedding": n_embedding,
        "n_blocks": n_blocks,
        "n_head": n_head,
        "dropout": dropout,
        "max_iters": max_iters,
        "learning_rate": learning_rate,
        "eval_every": eval_every,
        "eval_iters": eval_iters,
    }
)

print("WandB done")

# ====================================== DATA ======================================

# Dataset loading
file_path = "movie_reviews_dataset/all_reviews.csv"
dataSet = pd.read_csv(file_path)

print("Dataset loaded")

# Decreasing the size of the dataset to speed up the training process
# dataSet = dataSet.sample(frac=0.1, random_state=42)

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
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
dataSet['reviewText'] = dataSet['reviewText'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
)

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
tokenizer = Tokenizer(num_words=num_words, oov_token="<UNK>")  # <UNK> is the out-of-vocabulary token
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

n = int(0.9 * len(encoded_reviews))

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

# ====================================== GPT SENTIMENT ANALYSIS MODEL ======================================

# Class Head implementation
# The Head class is a single attention head in the multi-head attention mechanism
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        # The lower triangular matrix is used to mask future tokens in the attention mechanism
        # In classification, we do not need to mask future tokens
        # self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        # Relative positional encoding is used to capture the relative positions of tokens in the sequence
        self.rel_pos = RelativePositionalEncoding(head_size, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)
        v = self.value(x)  # [B, T, H]
        H = k.shape[-1]

        # Compute the relative positions between each pair of tokens
        positions = torch.arange(T, device=x.device)
        relative_position_matrix = positions.unsqueeze(1) - positions.unsqueeze(0)  # [T, T]
        
        # Retrieve the embeddings corresponding to these relative positions
        relative_position_embeddings = self.rel_pos.get_rel_embedding(relative_position_matrix)  # [T, T, H]
        
        # Content-based attention scores 
        content_score = q @ k.transpose(-2, -1)  # [B, T, T]
        
        # Attention scores based on relative position
        # Combine queries with relative positional embeddings
        # The einsum operation performs an intelligent tensor contraction
        relative_position_score = torch.einsum('bth,ijh->btij', q, relative_position_embeddings)  # [B, T, T]
        relative_position_score = relative_position_score.diagonal(dim1=-2, dim2=-1).transpose(-2, -1)  # [B, T, T]
        
        # Combine information from token and relative position scores
        weight = (content_score + relative_position_score) * H**-0.5
        
        # Softmax and dropout
        weight = F.softmax(weight, dim=-1)  # [B, T, T]
        weight = self.dropout(weight)
        
        # Value-weightghted attention
        h = weight @ v  # [B, T, T] @ [B, T, H] -> [B, T, H]

        return h

# Class MuliHeadAttention implementation
# The MuliHeadAttention class is a wrapper around multiple attention heads
class MuliHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.cat([h(x) for h in self.heads], dim=-1)
        h = self.proj(h)
        h = self.dropout(h)
        return h

# Class FeedForward implementation
# The FeedForward class is a feedforward neural network used in the transformer architecture
class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.GELU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
# Class Block implementation
# The Block class is a single transformer block that consists of multi-head attention and feedforward layers
class Block(nn.Module):
    def __init__(self, n_embedding, n_head):
        super().__init__()
        head_size = n_embedding // n_head
        self.sa = MuliHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        h = x
        h = h + self.sa(self.ln1(h))
        h = h + self.ff(self.ln2(h))
        return h

# Class RelativePositionalEncoding implementation
# The RelativePositionalEncoding class is used to encode relative positions in the input sequence    
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        # For a max_len length sequence, we need to represent
        # all possible relative distances: from -(max_len-1) to +(max_len-1)
        self.rel_embeddings = nn.Parameter(torch.randn(2*max_len-1, d_model) * 0.02)
        self.max_len = max_len
        
    def get_rel_embedding(self, relative_positions):
        shifted_positions = relative_positions + self.max_len - 1
        return self.rel_embeddings[shifted_positions]

# Class GPTSentimentAnalysisModel implementation
# The GPTSentimentAnalysisModel class is the main model for sentiment analysis
class GPTSentimentAnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()
        # token embedding table converts token indices to embeddings
        self.token_embedding_table = nn.Embedding(num_words, n_embedding)
        # No more position embedding table is used in this model
        # position embedding table converts position indices to embeddings
        # self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        # stack of transformer blocks to process the input
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embedding)
        # final layer normalization
        self.classifier = nn.Linear(n_embedding, 2)  # 2 classes : POSITIVE and NEGATIVE

    def forward(self, x):
        B, T = x.shape

        # Tokens (x) are converted to embeddings vectors (tok_emb)
        h = self.token_embedding_table(x)  # (B, T, C)
        # No position embeddings are used in this model
        # Position (0, 1, 2, ..., T-1) are converted to embeddings vectors (pos_emb)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        # The token embeddings and position embeddings are summed together to form the input to the transformer blocks
        # h = tok_emb + pos_emb  # (B, T, C)
        # The input is passed through the transformer blocks
        h = self.blocks(h)
        # The final layer normalization is applied to the output of the transformer blocks
        h = self.ln_f(h)
        # Pooling on the sequence dimension
        h = h.mean(dim=1)
        logits = self.classifier(h)  # (B, 2)
        return logits

# Loss function
# The classification_loss function computes the cross-entropy loss for the model's predictions
def classification_loss(model, x, targets):
    logits = model(x)  # (B, 2)
    loss = F.cross_entropy(logits, targets)
    return loss

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
            loss = criterion(logits, y)  # Calculer la perte
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

def calculate_accuracy(model, data, labels):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).sum().item()
        accuracy = correct / len(labels)
    return accuracy

# Model initialization
lm = GPTSentimentAnalysisModel().to(device)
optimizer = torch.optim.AdamW(lm.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("Model initialized")

# ====================================== TRAINING LOOP ======================================

print("Parameters:", sum(p.numel() for p in lm.parameters())/1e6, 'M parameters')

# Initialize the scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=max_iters)


best_validation_loss = float("inf")
patience_counter = 0
for i in range(max_iters + 1):
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
            "validation_accuracy": results['validation_accuracy'],
            "learning_rate": scheduler.get_last_lr()[0]        })

        lm.train()

        # Early stopping check
        if results['validation_loss'] < best_validation_loss:
            best_validation_loss = results['validation_loss']
            patience_counter = 0  # Reinitialize the patience counter
            # Save the best model
            torch.save(lm.state_dict(), f"{checkpoint}_best")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

            if (patience_counter >= patience):
                print("Early stopping triggered. Stopping training.")
                break

    # Training
    optimizer.zero_grad()
    x, y = get_batch(train_reviews, train_labels)
    loss = classification_loss(lm, x, y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Model save
    if i % save_every == 0:
        torch.save(lm.state_dict(), checkpoint)

print("Training done")

# ====================================== FINAL EVALUATION ======================================

# from sklearn.metrics import classification_report

# lm.eval()
# with torch.no_grad():
#     logits = lm(torch.stack(test_reviews))
#     predictions = torch.argmax(logits, dim=1)
#     print(classification_report(test_labels.cpu(), predictions.cpu(), target_names=["NEGATIVE", "POSITIVE"]))