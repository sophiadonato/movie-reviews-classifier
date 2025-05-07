import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import wandb
from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Whitespace

from torchvision import datasets, transforms, utils

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

vocab_n = 10000
sequence_length = 64
batch_size = 32
num_workers = 0

learning_rate = 1e-3
num_epochs = 50

model_name = "CNN_2Layer_20"
directory = "../vilour/CNN"
checkpoint = f"{directory}/{model_name}.pt"


input_features = sequence_length
output_features = 2

output_Features1 = 64
output_Features2 = 256
output_Features3 = 512

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
        "number of linear layers": number_of_linear_layers,

        "eval_every": eval_every,
        "part of the dataset": dataset_part,
    }
)

class SentimentDataset(data.Dataset):
    def __init__(self, root_dir, train=True):
        super().__init__()
        dataSet = pd.read_csv(root_dir)

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.enable_padding(length=sequence_length)
        tokenizer.enable_truncation(max_length=sequence_length)
        tokenizer_trainer = trainers.BpeTrainer(vocab_size=vocab_n)


        data = []
        for index, row in dataSet.iterrows():
            if not isinstance(row['reviewText'], str):
                continue
            if row['scoreSentiment'] == "POSITIVE":
                data.append((row['reviewText'], 1))
            elif row['scoreSentiment'] == "NEGATIVE":
                data.append((row['reviewText'], 0))

        datalen = round(len(data) * dataset_part)

        train_size = int(0.8 * datalen)
        test_size = datalen - train_size

        trainData = data[:train_size]
        textToTrainTO = []
        for text, label in trainData:
            textToTrainTO.append(text)

        for i in range(len(textToTrainTO)):
            if i > 10:
                break

        tokenizer.train_from_iterator(textToTrainTO, trainer=tokenizer_trainer)
        self.tokenizer = tokenizer

        self.data = data[:train_size] if train else data[train_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer.encode(text)

        return torch.tensor(encoding.ids, dtype=torch.float), torch.tensor(label, dtype=torch.long)



for i in range(number_of_conv_layers):
    input_features = (input_features - 2) // 2


class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(1, output_Features1, 3),
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
        )

network = CNN()

device = torch.device(device)
network.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

my_trainset = SentimentDataset('rotten_tomatoes_movie_reviews.csv', train=True)
my_trainloader = data.DataLoader(my_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

my_testset = SentimentDataset('rotten_tomatoes_movie_reviews.csv', train=False)
my_testloader = data.DataLoader(my_testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

print(f"Parameters: {sum(p.numel() for p in network.parameters())/1e6} M parameters")
def train(num_epochs):
    train_avg_loss = []
    test_avg_loss = []
    test_accuracy = []

    for i in range(num_epochs + 1):
        print(f"Epoch number : {i+1}")
        train_losses = []
        test_losses = []

        for x, y in my_trainloader:
            x, y = x.unsqueeze(1).to(device), y.to(device)

            pred = network(x)
            loss = criterion(pred, y)
            train_losses.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct = 0
            for x, y in my_testloader:
                x, y = x.unsqueeze(1).to(device), y.to(device)

                pred = network(x)
                loss = criterion(pred, y)
                test_losses.append(loss.detach())

                y_pred = pred.argmax(dim=1)
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

