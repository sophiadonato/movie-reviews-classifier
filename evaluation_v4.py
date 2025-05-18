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
train_set, test_set = torchdata.random_split(dataset, [int(0*len(dataset)), len(dataset)-int(0*len(dataset))])
#trainloader = torchdata.DataLoader(train_set, batch_size=batch_size, shuffle=True)
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
wandb.init(project="evaluation-review-classifier", name="evaluation-v4-longertext")
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