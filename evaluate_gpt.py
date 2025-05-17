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

#Hyperparameters
batch_size = 32
block_size = 512
n_embedding = 512
# Having a minimum large nb of blocks and heads is important to capture
# complex patterns in the data
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

#Data extraction
data = pd.read_csv("./movie_reviews_dataset/rotten_tomatoes_movie_reviews.csv")
data = data.dropna(subset=["reviewText"])

sentiment_mapping = {'POSITIVE' : 0, 'NEGATIVE' : 1}
data["scoreSentiment"] = data["scoreSentiment"].map(sentiment_mapping).to_numpy()

nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

#Data preprocessing
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

data["reviewText"] = data["reviewText"].apply(clean_text)

X = data["reviewText"].to_numpy()
y = data["scoreSentiment"].to_numpy()

#Tokenization and padding
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

X_padded = pad_sequences(X_sequences, padding='post', maxlen=max_seq_len)

#Datasets ready
X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = torchdata.TensorDataset(X_tensor, y_tensor)
train_set, test_set = torchdata.random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])
trainloader = torchdata.DataLoader(train_set, batch_size=batch_size, shuffle=True)
testloader = torchdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#Model
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

model_name = "sentiment-analysis-gpt"
directory = "./checkpoints"
checkpoint_path = f"{directory}/{model_name}.pt"
net = GPTSentimentAnalysisModel().to(device)
net.load_state_dict(torch.load(checkpoint_path))
net = net.to(device)
net.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for x,y in testloader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        predictions = torch.argmax(logits, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(y.int().cpu().numpy())

# Confusion matrix
wandb.init(project="evaluation-review-classifier", name="GPT_2_evaluation")
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=["POSITIVE", "NEGATIVE"]
    )
})


# New reviews evaluation
new_reviews = ["This movie is amazing !", "I didn't like ThunderBolts, it was really shitty", "I am a bit confused about this movie, realisation is super great but acting is terrible..", "My eyes physically hurt from watching this movie", "Best day ever ! Realisation, acting, costumes, .. everything was perfect",
            "This film radiates charm from start to finish. The performances are sincere, the cinematography is stunning, and the pacing feels just right. There’s an emotional core that resonates deeply, even in quieter moments. While the plot isn’t groundbreaking, its execution elevates it beyond its genre peers. The dialogue is witty without being forced, and the direction feels confident. It’s rare to see a movie balance warmth and melancholy this well. The ending lingers in your thoughts in the best way. If you’re looking for a heartfelt story told with care, this one is absolutely worth your time.",
            "Despite a strong cast and striking visuals, the movie fails to leave a lasting impression. The story meanders without urgency, characters remain underdeveloped, and tonal shifts undercut any emotional weight. There are moments of intrigue, but they never fully blossom into anything meaningful. Some might appreciate the film’s moodiness, but its sluggish pace and vague narrative make it a challenge to sit through. It feels like the director had something important to say but lost clarity along the way. For a film that aims high, it sadly misses the mark more often than it hits.",
            "It’s hard to pin down exactly how I feel about this film. It has moments of brilliance—especially in its visuals and soundtrack—but also stretches that feel unnecessarily drawn out or oddly disjointed. Performances are competent, if not particularly memorable. There’s clearly ambition behind the scenes, but whether it all comes together depends on your expectations. It’s not bad, not exactly great either—more of an experience than a traditional narrative. You might walk away inspired or confused, or maybe both. It definitely tries something different, and for that alone, it might be worth a curious viewing.",
            "A loud, messy spectacle that confuses chaos with entertainment. The plot is an incoherent mash of clichés, the dialogue is cringeworthy, and even the action sequences feel repetitive. Whatever potential the premise had is buried beneath bad editing and flat performances. The humor misses more often than it hits, and the emotional beats feel unearned. It’s the kind of movie where you keep checking the time, hoping it’s almost over. Some viewers might enjoy the over-the-top energy, but most will likely leave feeling disappointed and a little exhausted. There’s just not much here to hold onto.",
            "What a pleasant surprise. This quiet, understated film manages to capture something deeply human without relying on melodrama or spectacle. The performances are subtle, the direction is assured, and the writing feels authentic. Nothing is over-explained, yet everything lands emotionally. It’s the kind of story that sneaks up on you and stays in your mind for days afterward. The cinematography isn’t flashy, but it perfectly suits the mood. It may not be for everyone—especially those expecting big twists—but if you’re in the mood for a thoughtful, character-driven story, this one delivers with quiet grace.",
            "This film certainly exists. It has actors, it has a script, and things happen. Whether any of it adds up to something worth watching is debatable. There are brief flashes of creativity, but they’re surrounded by long stretches of mediocrity. The tone shifts awkwardly, and the ending feels more like a shrug than a statement. It’s not offensively bad, just frustratingly bland. You won’t walk away angry—just wondering if your time could’ve been better spent elsewhere. Still, there’s a kind of strange charm in its odd moments, so maybe it’s not a total loss?",
            "An absolute gem. From the opening scene, the film draws you into its world with confidence and style. Every frame feels deliberate, the performances are magnetic, and the story unfolds with perfect pacing. Emotional without being manipulative, smart without being smug—it hits a rare balance. The score complements the tone beautifully, and the finale ties everything together with grace. It’s the kind of movie you want to rewatch immediately just to catch what you missed. A masterclass in storytelling that reminds you why cinema matters. Don’t miss it.",
            "I wanted to like this movie. The premise was intriguing, and the opening scenes promised something bold. But somewhere along the way, it lost momentum. Characters make baffling choices, subplots go nowhere, and the pacing drags. There are still interesting elements—a strong performance here, a striking shot there—but they feel isolated, like pieces of a better film that never came together. It’s not terrible, just deeply uneven. By the time the credits roll, it’s hard to feel anything but disappointment. A missed opportunity.",
            "This film defies easy categorization. It’s neither a straightforward drama nor a full-fledged comedy, but something that hovers in between. Some will find that exciting, others will find it frustrating. There are long, quiet scenes that invite reflection—and just as many where nothing seems to happen. It’s well-shot and competently acted, but hard to describe or summarize. Does that make it profound or just pretentious? It probably depends on your mood. One thing’s for sure: it won’t please everyone. And maybe that’s the point.",
            "A joyful rollercoaster of a film that embraces its absurdity and runs with it. Sure, it’s ridiculous and over-the-top, but that’s exactly the charm. The cast is clearly having a blast, the script is snappy, and the energy never lets up. It knows what it is and doesn’t try to be more than that. The result is pure fun—flawed, yes, but undeniably entertaining. If you’re in the mood for something light and fast-paced, this might hit the spot. Just don’t expect depth or subtlety, and you’ll have a great time."]
new_reviews_cleaned = [clean_text(review) for review in new_reviews]
new_reviews_seq = tokenizer.texts_to_sequences(new_reviews_cleaned)
new_reviews_padded = pad_sequences(new_reviews_seq, padding='post', maxlen = max_seq_len)
new_reviews_tensor = torch.tensor(new_reviews_padded, dtype=torch.long).to(device)

with torch.no_grad():
    logits = net(new_reviews_tensor)
    predictions = torch.argmax(logits, dim=1)
    pred_labels = predictions.cpu().numpy()

table = wandb.Table(columns=["Reviews", "Prediction"])
for new_review, pred_label in zip(new_reviews, pred_labels):
    table.add_data(new_review, "POSITIVE" if pred_label == 0 else "NEGATIVE") 
wandb.log({"Reviews predicition" : table})