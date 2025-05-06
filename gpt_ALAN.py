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

# ====================================== GPU DETECTION ======================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(100)

# ====================================== HYPERPARAMETERS ======================================

batch_size = 16
block_size = 256
n_embedding = 128
n_blocks = 4
n_head = 2
dropout = 0.5
max_iters = 5000
learning_rate = 3e-4
eval_every = 500
eval_iters = 200
save_every = 10000

# ====================================== CHECKPOINTS ======================================

model_name = "sentiment-analysis-gpt"
directory = "./checkpoints"
os.makedirs(directory, exist_ok=True)
checkpoint = f"{directory}/{model_name}.pt"

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

# ====================================== DATA ======================================

# Dataset loading
file_path = "movie_reviews_dataset/all_reviews.csv"
dataSet = pd.read_csv(file_path)

# Decreasing the size of the dataset to speed up the training process
dataSet = dataSet.sample(frac=0.1, random_state=42)

# DATASET PREPROCESSING

# 1. Replace POSITIVE and NEGATIVE with 1 and 0
dataSet['scoreSentiment'] = dataSet['scoreSentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})
# 2. Removes punctuation and special characters
dataSet['reviewText'] = dataSet['reviewText'].str.replace('[^\w\s]', '', regex=True)

# 3. Removes stop words which do not add value to the analysis
# Stopwords download
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

dataSet['reviewText'] = dataSet['reviewText'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# Preprocessing ideas 2 and 3 comes from the following link:
# https://medium.com/@felixmutai/tokenization-in-sentiment-analysis-cd386a08fab

# Dataset conversion to lists in order to be used by the tokenizer 
# Initially, datasets are in pandas DataFrame format and need to be converted to lists for tokenization
reviews = dataSet['reviewText'].tolist()
labels = dataSet['scoreSentiment'].tolist()

# TOKENIZATION 

tokenizer = ByteLevelBPETokenizer()
tokenizer.train([file_path])

# Add a special token <PAD> to the tokenizer vocabulary
tokenizer.add_special_tokens(["<PAD>"])
pad_token_id = tokenizer.token_to_id("<PAD>")

# Convert a review to a list of integers
encode = lambda s: tokenizer.encode(s).ids

# Convert a list of integers to a review
decode = lambda l: tokenizer.decode(l)

# Encoding of reviews and labels
# During encoding,  each sequence is converted into a torch.LongTenser in order to be used by the model which only accepts torch tensors
encoded_reviews = [torch.tensor(encode(review), dtype=torch.long).to(device) for review in reviews]
encoded_labels = torch.tensor(labels, dtype=torch.long).to(device)

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

n = int(0.9 * len(reviews))

# Define the maximum length for padding
# Limit the maximum length of the sequences to a fixed size (block_size) for training
max_length = min(block_size, max(len(seq) for seq in encoded_reviews))

# Apply padding to the sequences
train_reviews = pad_sequences(encoded_reviews[:n], max_length, pad_token_id)
test_reviews = pad_sequences(encoded_reviews[n:], max_length, pad_token_id)

train_labels = encoded_labels[:n]
test_labels = encoded_labels[n:]

# DATA LOADER

def get_batch(data, labels):
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i] for i in ix])
    y = torch.stack([labels[i] for i in ix])
    x, y = x.to(device), y.to(device)
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
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)
        H = k.shape[-1]

        # compute attention scores (affinities)
        # attention scores are the measure of how much focus each token should have on other tokens in the sequence
        # attention scores are computed by taking the dot product of the query and key vectors
        wei = q @ k.transpose(-1, -2) * H**-0.5 # (B, T, H) @ (B, H, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # weighted aggregation of the values
        v = self.value(x) # (B, T, H)
        h = wei @ v # (B, T, T) @ (B, T, H) -> (B, T, H)

        return h

# Class MuliHeadAttention implementation
# The MuliHeadAttention class is a wrapper around multiple attention heads
class MuliHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embedding, n_embedding)
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

# Class GPTSentimentAnalysisModel implementation
# The GPTSentimentAnalysisModel class is the main model for sentiment analysis
class GPTSentimentAnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()
        # token embedding table converts token indices to embeddings
        self.token_embedding_table = nn.Embedding(len(tokenizer.get_vocab()), n_embedding)
        # position embedding table converts position indices to embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        # stack of transformer blocks to process the input
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embedding)
        # final layer normalization
        self.classifier = nn.Linear(n_embedding, 2)  # 2 classes : POSITIVE and NEGATIVE

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        h = tok_emb + pos_emb  # (B, T, C)
        h = self.blocks(h)
        h = self.ln_f(h)
        h = h.mean(dim=1)  # Pooling on the sequence dimension
        logits = self.classifier(h)  # (B, 2)
        return logits

# Loss function
# The classification_loss function computes the cross-entropy loss for the model's predictions
def classification_loss(model, x, targets):
    logits = model(x)  # (B, 2)
    loss = F.cross_entropy(logits, targets)
    return loss

# Model initialization
lm = GPTSentimentAnalysisModel().to(device)
optimizer = torch.optim.AdamW(lm.parameters(), lr=learning_rate)

# ====================================== TRAINING LOOP ======================================

for i in range(max_iters + 1):
    if i % eval_every == 0:
        # Evaluation on the test set
        with torch.no_grad():
            train_loss = classification_loss(lm, train_reviews, train_labels)
            val_loss = classification_loss(lm, test_reviews, test_labels)

        print(f"Step {i}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        wandb.log({
            "step": i,
            "train_loss": train_loss.item(),
            "validation_loss": val_loss.item()
        })

        lm.train()

    # Training
    optimizer.zero_grad()
    x, y = get_batch(train_reviews, train_labels)
    loss = classification_loss(lm, x, y)
    loss.backward()
    optimizer.step()

    # Model save
    if i % save_every == 0:
        torch.save(lm.state_dict(), checkpoint)

# ====================================== FINAL EVALUATION ======================================

# from sklearn.metrics import classification_report

# lm.eval()
# with torch.no_grad():
#     logits = lm(torch.stack(test_reviews))
#     predictions = torch.argmax(logits, dim=1)
#     print(classification_report(test_labels.cpu(), predictions.cpu(), target_names=["NEGATIVE", "POSITIVE"]))