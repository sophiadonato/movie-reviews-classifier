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
testloader = torchdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)

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
    
checkpoint_path = "C:/Users/sdona/Documents/M1Q2_DeepLearning/projet/checkpoints/simple-nn-v4.pt"
net = SimpleNNTextClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, input_len=max_sequence_length)
net.load_state_dict(torch.load(checkpoint_path))
net = net.to(device)
net.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for x,y in testloader:
        print("iterating")
        x, y = x.to(device), y.to(device)
        logits = net(x)
        probs = torch.sigmoid(logits)
        pred_class = (probs>0.5).int().cpu().numpy()
        all_preds.extend(pred_class)
        all_labels.extend(y.int().cpu().numpy())

# Confusion matrix
wandb.init(project="evaluation-review-classifier", name="evaluation-v4")
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=["POSITIVE", "NEGATIVE"]
    )
})


# New reviews evaluation
new_reviews = ["This movie is amazing !", "I didn't like ThunderBolts, it was really shitty", "I am a bit confused about this movie, realisation is super great but acting is terrible..", "My eyes physically hurt from watching this movie", "Best day ever ! Realisation, acting, costumes, .. everything was perfect"]
new_reviews_cleaned = [clean_text(review) for review in new_reviews]
new_reviews_seq = tokenizer.texts_to_sequences(new_reviews_cleaned)
new_reviews_padded = pad_sequences(new_reviews_seq, padding='post', maxlen = max_sequence_length)
new_reviews_tensor = torch.tensor(new_reviews_padded, dtype=torch.long).to(device)

with torch.no_grad():
    preds = net(new_reviews_tensor)
    pred_labels = torch.sigmoid(preds) > 0.5
    pred_labels = pred_labels.cpu().numpy()

table = wandb.Table(columns = ["Reviews", "Prediction"])
for new_review, pred_label in zip(new_reviews, pred_labels):
    table.add_data(new_review, "POSITIVE" if pred_label == 0 else "NEGATIVE") 
wandb.log({"Reviews predicition" : table})