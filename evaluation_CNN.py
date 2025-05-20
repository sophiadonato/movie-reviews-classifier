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

# ====================================== HYPERPARAMETERS ======================================
vocab_n = 10000
sequence_length = 100
batch_size = 32
num_workers = 0

learning_rate = 1e-3
num_epochs = 200

input_features = sequence_length
output_features = 2

output_Features1 = 64
output_Features2 = 256
output_Features3 = 512
output_Features4 = 1024
kernel_size = 3

output_Features_final = 1024

output_linear1 = 512
output_linear2 = 256
output_linear3 = 128
output_linear4 = 64

number_of_conv_layers = 4
number_of_linear_layers = 4

eval_every = 1
save_every = 10

dataset_part = 1
dropout = 0.3

for i in range(number_of_conv_layers):
    input_features = (input_features - kernel_size + 1) // 2


#Data extraction
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

dataset = torchdata.TensorDataset(encoded_reviews, encoded_labels)
train_set, test_set = torchdata.random_split(dataset, [int(0*len(dataset)), len(dataset)-int(0*len(dataset))])
#trainloader = torchdata.DataLoader(train_set, batch_size=batch_size, shuffle=True)
testloader = torchdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#Model
class CNN(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.embedding1 = nn.Embedding(vocab_n, 15)
        self.conv1d1 = nn.Conv1d(15, output_Features1, kernel_size)
        self.reLU1 = nn.ReLU()
        self.maxPool1d1 = nn.MaxPool1d(2, stride=2)
        self.conv1d2 = nn.Conv1d(output_Features1, output_Features2, kernel_size)
        self.reLU2 = nn.ReLU()
        self.maxPool1d2 = nn.MaxPool1d(2, stride=2)
        self.conv1d3 = nn.Conv1d(output_Features2, output_Features3, kernel_size)
        self.reLU3 = nn.ReLU()
        self.maxPool1d3 = nn.MaxPool1d(2, stride=2)
        self.conv1d4 = nn.Conv1d(output_Features3, output_Features4, kernel_size)
        self.reLUConv4 = nn.ReLU()
        self.maxPool1d4 = nn.MaxPool1d(2, stride=2)
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(input_features * output_Features_final, output_linear1)
        self.reLU4 = nn.ReLU()
        self.linear2 = nn.Linear(output_linear1, output_linear2)
        self.reLU5 = nn.ReLU()
        self.linear3 = nn.Linear(output_linear2, output_linear3)
        self.reLU6 = nn.ReLU()
        self.linear4 = nn.Linear(output_linear3, output_linear4)
        self.relU7 = nn.ReLU()
        self.linear5 = nn.Linear(output_linear4, output_features)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.embedding1(x)
        x = x.view(x.shape[0], 15, 100)
        x = self.conv1d1(x)
        x = self.reLU1(x)
        x = self.maxPool1d1(x)
        x = self.conv1d2(x)
        x = self.reLU2(x)
        x = self.maxPool1d2(x)
        x = self.conv1d3(x)
        x = self.reLU3(x)
        x = self.maxPool1d3(x)
        x = self.conv1d4(x)
        x = self.reLUConv4(x)
        x = self.maxPool1d4(x)
        x = self.flatten1(x)

        x = self.linear1(x)
        x = self.reLU4(x)
        x = self.linear2(x)
        x = self.reLU5(x)
        x = self.linear3(x)
        x = self.reLU6(x)
        x = self.linear4(x)
        x = self.relU7(x)
        x = self.linear5(x)
        x = self.dropout(x)
        return x
    
checkpoint_path = "../vilour/CNN/CNN_evaluation.pt"
net = CNN()
net.load_state_dict(torch.load(checkpoint_path, map_location=device))
net = net.to(device)
net.eval()

wandb.init(project="evaluation-review-CNN")

all_preds, all_labels = [], []
with torch.no_grad():
    for x,y in testloader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        pred_class = torch.argmax(logits, dim=1)
        all_preds.extend(pred_class.int().cpu().numpy())
        all_labels.extend(y.int().cpu().numpy())

print("Unique preds:", set(all_preds))      # Should be {0, 1}
print("Unique labels:", set(all_labels))    # Should be {0, 1}

number_correct = 0
for x, y in zip(all_preds, all_labels):
    if x == y:
        number_correct += 1

print(f"The accuracy is {number_correct/len(all_preds)}")

# Confusion matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=["NEGATIVE", "POSITIVE"]
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

# 2. Removes punctuation and special characters
new_reviews = new_reviews.str.replace('[^\w\s]', '', regex=True)

# 3. Removes stop words which do not add value to the analysis
# Stopwords download
# nltk.download("stopwords")
# stop_words = set(stopwords.words('english'))

# new_reviews = new_reviews.apply(
#     lambda x: ' '.join([word for word in x.split() if word not in stop_words])
# )
# not or never being stop words and having a strong impact on the sentiment,
# stop words should not been removed

# 4. Lemmatization to reduce words to their base form
# nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()
# new_reviews = new_reviews.apply(
#     lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
# )

# 5. Convert to lowercase
new_reviews = new_reviews.str.lower()

# 6. Remove extra spaces
new_reviews_cleaned = new_reviews.str.replace(r'\s+', ' ', regex=True).str.strip()


new_reviews_seq = tokenizer.texts_to_sequences(new_reviews_cleaned)
new_reviews_padded = pad_sequences(new_reviews_seq, padding='post', maxlen = sequence_length)
new_reviews_tensor = torch.tensor(new_reviews_padded, dtype=torch.long).to(device)

with torch.no_grad():
    preds = net(new_reviews_tensor)
    pred_labels = torch.argmax(preds, dim=1).cpu().numpy()

table = wandb.Table(columns = ["Reviews", "Prediction"])
for new_review, pred_label in zip(new_reviews, pred_labels):
    table.add_data(new_review, "POSITIVE" if pred_label == 1 else "NEGATIVE") 
wandb.log({"Reviews predicition" : table})

