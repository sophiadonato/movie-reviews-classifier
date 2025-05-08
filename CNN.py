import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import wandb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from torchvision import datasets, transforms, utils

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

vocab_n = 10000
sequence_length = 100
batch_size = 32
num_workers = 0

learning_rate = 1e-3
num_epochs = 50

model_name = "CNN_2Layer_20"
directory = "../vilour/CNN"
checkpoint = f"{directory}/{model_name}.pt"


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

dataSet = pd.read_csv(root_dir)
dataSet = dataSet.dropna(subset=["reviewText"])

sentiment_mapping = {'POSITIVE': 0, 'NEGATIVE': 1}
dataSet["scoreSentiment"] = dataSet["scoreSentiment"].map(sentiment_mapping).to_numpy()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

dataSet["reviewText"] = dataSet["reviewText"].apply(clean_text)

X = dataSet["reviewText"].to_numpy()
Y = dataSet["scoreSentiment"].to_numpy()

tokenizer = Tokenizer(num_words=vocab_n)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequence(X)
vocab_size = vocab_n + 1

X_padded = pad_sequences(X_sequences, padding='post', maxlen=sequence_length)

datalen = round(len(dataSet) * dataset_part)

train_size = int(0.8 * datalen)
test_size = datalen - train_size

X_tensor = torch.tensor(X_padded, dtype=torch.long)
y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = dataSet.TensorDataset(X_tensor, y_tensor)
train_set, test_set = data.random_split(dataset, [train_size, test_size])
my_trainloader = dataSet.DataLoader(train_set, batch_size=batch_size, shuffle=True)
my_testloader = dataSet.DataLoader(test_set, batch_size=batch_size, shuffle=False)



for i in range(number_of_conv_layers):
    input_features = (input_features - 2) // 2


class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Embedding(vocab_size, 15),
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

criterion = nn.BCELoss()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

print(f"Parameters: {sum(p.numel() for p in network.parameters())/1e6} M parameters")
def train(num_epochs):
    train_avg_loss = []
    test_avg_loss = []
    test_accuracy = []

    for i in range(num_epochs + 1):
        print(f"Epoch number : {i+1}")
        train_losses = []
        test_losses = []

        network.train()

        for x, y in my_trainloader:
            pred = network(x)
            loss = criterion(pred, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct = 0
            for x, y in my_testloader:
                pred = network(x)
                loss = criterion(pred, y)
                test_losses.append(loss.detach())

                y_pred = (pred > 0.5).float()
                correct = correct + (y_pred == y).sum()

            accuracy = correct / len(my_testset)

        print(f"The accuracy is {accuracy}")
        test_accuracy.append(accuracy.cpu())
        train_avg_loss.append(torch.mean(torch.tensor(train_losses)))
        test_avg_loss.append(torch.mean(torch.tensor(test_losses)))

        if i % eval_every == 0:
            print(f"step {i}: train loss {train_avg_loss}, validation loss {test_avg_loss} and test accuracy = {test_accuracy}")
            wandb.log({
                "step": i,
                "train loss": train_avg_loss[i],
                "validation loss": test_avg_loss[i],
                "test accuracy": test_accuracy[i],
            })
        if i % save_every == 0:
            torch.save(network.state_dict(), checkpoint)


    return train_avg_loss, test_avg_loss, test_accuracy


train_avg_loss, test_avg_loss, test_accuracy = train(num_epochs)


fig, axs = plt.subplots(3, 1)

axs[0].set_title('Training Average Loss')
axs[0].plot(train_avg_loss)
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss ')

axs[1].set_title('Testing Average Loss')
axs[1].plot(test_avg_loss)
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss ')

axs[2].set_title('Testing Accuracy')
axs[2].plot(test_accuracy)
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()

