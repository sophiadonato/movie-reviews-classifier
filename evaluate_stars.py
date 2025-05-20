import pandas as pd
import numpy as np 
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import matplotlib.pyplot as plt
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import nltk
from nltk.stem import WordNetLemmatizer

#GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
torch.manual_seed(100)

#Hyperparameters (similaires à GPT_stars.py)
batch_size = 128
block_size = 128
n_embedding = 512
n_blocks = 12
n_head = 8
dropout = 0.2
max_iters = 10000
learning_rate = 1e-4
eval_every = 500
eval_iters = 200
save_every = 1000
num_words = 50000
max_seq_len = 128

# Fonction reprise de GPT_stars.py pour normaliser les scores
def normalize_score(score):
    if pd.isna(score) or score == '':
        return None
    
    # Conversion into format X/Y
    fraction_match = re.match(r'(\d+(\.\d+)?)/(\d+(\.\d+)?)', str(score))
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(3))
        if denominator == 0:
            return None
        normalized = (numerator / denominator) * 5
        return round(normalized)
    
    # Convert score into format X.Y
    try:
        numeric_score = float(score)
        # Determine the scale based on the value
        if numeric_score <= 5:  # Supposed to be on a 5-point scale
            return round(numeric_score)
        elif numeric_score <= 10:  # Supposed to be on a 10-point scale
            return round(numeric_score / 2)
        else:  # Unknown scale, return None
            return None
    except ValueError:
        # Letter, return None
        return None

#Data extraction
data = pd.read_csv("./movie_reviews_dataset/rotten_tomatoes_movie_reviews.csv")
data = data.dropna(subset=["reviewText"])

# DATASET PREPROCESSING
print("Starting data preprocessing...")

# 1. Normalize scores
data['star_rating'] = data['originalScore'].apply(normalize_score)
# Filter non valid score
data = data.dropna(subset=['star_rating'])
# Convert score to int
data['star_rating'] = data['star_rating'].astype(int)
# Clip the star_rating to be between 0 and 5
data['star_rating'] = data['star_rating'].clip(0, 5)

# 2. Removes punctuation and special characters
data['reviewText'] = data['reviewText'].str.replace('[^\w\s]', '', regex=True)

# 3. Lemmatization to reduce words to their base form
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
data['reviewText'] = data['reviewText'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
)

# 4. Convert to lowercase
data['reviewText'] = data['reviewText'].str.lower()

# 5. Remove extra spaces
data['reviewText'] = data['reviewText'].str.replace(r'\s+', ' ', regex=True).str.strip()

# 6. Remove empty reviews
data = data[data['reviewText'].str.strip() != '']

print("Dataset preprocessing complete")

X = data["reviewText"].to_numpy()
y = data["star_rating"].to_numpy()

#Tokenization and padding
tokenizer = Tokenizer(num_words=num_words, oov_token="<UNK>")
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

X_padded = pad_sequences(X_sequences, padding='post', maxlen=max_seq_len)

#Datasets ready
X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = torchdata.TensorDataset(X_tensor, y_tensor)
test_set = torchdata.Subset(dataset, range(min(10000, len(dataset))))
testloader = torchdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#Model classes copied from GPT_stars.py

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        self.rel_embeddings = nn.Parameter(torch.randn(2*max_len-1, d_model) * 0.02)
        self.max_len = max_len
        
    def get_rel_embedding(self, relative_positions):
        shifted_positions = relative_positions + self.max_len - 1
        return self.rel_embeddings[shifted_positions]

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.rel_pos = RelativePositionalEncoding(head_size, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        H = k.shape[-1]

        positions = torch.arange(T, device=x.device)
        relative_position_matrix = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_position_embeddings = self.rel_pos.get_rel_embedding(relative_position_matrix)
        
        content_score = q @ k.transpose(-2, -1)
        relative_position_score = torch.einsum('bth,ijh->btij', q, relative_position_embeddings)
        relative_position_score = relative_position_score.diagonal(dim1=-2, dim2=-1).transpose(-2, -1)
        
        weight = (content_score + relative_position_score) * H**-0.5
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        
        h = weight @ v
        return h

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

# Modèle adapté pour la prédiction d'étoiles (0-5)
class GPTStarPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_words, n_embedding)
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(n_embedding)
        self.classifier = nn.Linear(n_embedding, 6)  # 6 classes pour 0-5 étoiles

    def forward(self, x):
        B, T = x.shape
        h = self.token_embedding_table(x)
        h = self.blocks(h)
        h = self.ln_f(h)
        h = h.mean(dim=1)  # Pooling sur dimension séquence
        logits = self.classifier(h)
        return logits

# Chargement du modèle
model_name = "sentiment-analysis-gpt-star-model"
directory = "./checkpoints"
best_checkpoint_path = f"{directory}/{model_name}_best.pt"

print(f"Loading model from {best_checkpoint_path}...")
checkpoint_data = torch.load(best_checkpoint_path)
net = GPTStarPredictionModel().to(device)
net.load_state_dict(checkpoint_data['model'])
net.eval()

print(f"Loaded checkpoint from iteration {checkpoint_data['iteration']}")
print(f"Best validation loss: {checkpoint_data['best_val_loss']}")
print(f"Best validation accuracy: {checkpoint_data['best_validation_accuracy']}")

# Évaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        predictions = torch.argmax(logits, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Confusion matrix
wandb.init(project="evaluation-review-classifier", name="GPT_star_evaluation")
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=["0 star", "1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    )
})

# New reviews evaluation
new_reviews = ["This movie is amazing !", "I didn't like ThunderBolts, it was really shitty", 
               "I am a bit confused about this movie, realisation is super great but acting is terrible..", 
               "My eyes physically hurt from watching this movie", 
               "Best day ever ! Realisation, acting, costumes, .. everything was perfect",
               "This film radiates charm from start to finish. The performances are sincere, the cinematography is stunning.",
               "Despite a strong cast and striking visuals, the movie fails to leave a lasting impression.",
               "It's hard to pin down exactly how I feel about this film. It has moments of brilliance—especially in its visuals."]

# Prétraitement des nouvelles critiques
new_reviews_processed = []
for review in new_reviews:
    # 1. Suppression des ponctuations et caractères spéciaux
    processed = re.sub(r'[^\w\s]', '', review)
    
    # 2. Lemmatisation 
    processed = ' '.join([lemmatizer.lemmatize(word) for word in processed.split()])
    
    # 3. Conversion en minuscules
    processed = processed.lower()
    
    # 4. Suppression des espaces en trop
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    new_reviews_processed.append(processed)

# Tokenisation et padding
new_reviews_seq = tokenizer.texts_to_sequences(new_reviews_processed)
new_reviews_padded = pad_sequences(new_reviews_seq, padding='post', maxlen=max_seq_len)
new_reviews_tensor = torch.tensor(new_reviews_padded, dtype=torch.long).to(device)

# Prédiction
with torch.no_grad():
    logits = net(new_reviews_tensor)
    predictions = torch.argmax(logits, dim=1)
    pred_labels = predictions.cpu().numpy()

# Affichage des résultats avec WandB
table = wandb.Table(columns=["Reviews", "Star Rating"])
for new_review, pred_label in zip(new_reviews, pred_labels):
    table.add_data(new_review, f"{pred_label} star{'s' if pred_label != 1 else ''}")
wandb.log({"Reviews Star Rating": table})

print("Evaluation complete!")