# Added dropout, decreased learning rate, added regularization and divided by 2 nb of epochs to avoid overfitting 
# Reduced batch size such that faster learning
import pandas as pd
import numpy as np 
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import matplotlib.pyplot as plt
import wandb

#GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
torch.manual_seed(100)

#Hyperparameters
batch_size = 16
max_iters = 64
learning_rate = 0.0001
num_words = 10000
vocab_size = num_words+1
max_sequence_length = 78 #computed by me in ipynb
embedding_dim = 16
eval_every = 2

#Wandb
model_name = "simple-nn-v4"
directory = "C:/Users/sdona/Documents/M1Q2_DeepLearning/projet/checkpoints"
checkpoint = f"{directory}/{model_name}.pt"

wandb.init(
    project = "review-classifier",
    config={
        "model" : model_name,
        "batch_size" :batch_size,
        "n_embedding" : embedding_dim,
        "max_iters " : max_iters,
        "learning_rate" :learning_rate,
        "eval_every" : eval_every,
    }
)

#Data extraction
data = pd.read_csv("../rotten_tomatoes_movie_reviews.csv")
data = data.dropna(subset=["reviewText"])

sentiment_mapping = {'POSITIVE' : 0, 'NEGATIVE' : 1}
data["scoreSentiment"] = data["scoreSentiment"].map(sentiment_mapping).to_numpy()

#Data preprocessing
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

data["reviewText"] = data["reviewText"].apply(clean_text)

X = data["reviewText"].to_numpy()
y = data["scoreSentiment"].to_numpy()

#Tokenization and padding
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

X_padded = pad_sequences(X_sequences, padding='post', maxlen=max_sequence_length)

#Datasets ready
X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = torchdata.TensorDataset(X_tensor, y_tensor)
train_set, test_set = torchdata.random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])
trainloader = torchdata.DataLoader(train_set, batch_size=batch_size, shuffle=True)
validationloader = torchdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#Model
class SimpleNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_len):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lin1 = nn.Linear(embedding_dim, 32)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.embedding(x) #(batch, seq_len, emb_dim)
        x = x.mean(dim=1) #(batch size, emb_dim)
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.lin3(x)
        return x.squeeze(1)

# Training loop
net = SimpleNNTextClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, input_len=max_sequence_length)
net = net.to(device)
print("Parameters:", sum(p.numel() for p in net.parameters())/1e3, 'K parameters')

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

@torch.no_grad()
def evaluate(net):
    net.eval()
    out = {}
    for split, loader in [("train", trainloader), ("validation", validationloader)]:
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for x, y in loader:
            x,y = x.to(device), y.to(device)
            net_output = net(x)
            pred = torch.sigmoid(net_output)
            loss = criterion(net_output, y)
            total_loss += loss.item()
            pred_class = (pred>0.5).float()
            total_correct += (pred_class == y).float().sum().item()
            total_samples += y.size(0)

        out[f"{split}_loss"] = total_loss/len(loader)
        out[f"{split}_accuracy"] = total_correct/total_samples
        
    net.train()
    return out

for i in range(max_iters + 1):
    print(i)
    if i%eval_every == 0:
        out = evaluate(net)
        print(f"Step {i} : Train accuracy {out['train_accuracy']:.4f}, Validation accuracy {out['validation_accuracy']:.4f}")

        wandb.log({
            "step" : i,
            "train loss" : out["train_loss"],
            "train accuracy" : out["train_accuracy"],
            "validation loss" : out["validation_loss"],
            "validation accuracy" : out["validation_accuracy"]
        })

    net.train()
    for batch_X, batch_y in trainloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        pred = net(batch_X)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save model
    torch.save(net.state_dict(), checkpoint)