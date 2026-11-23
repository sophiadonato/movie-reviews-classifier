
# movie-reviews-classifier

Course project for **INFO8010 – Deep Learning** (University of Liège).

**Authors:** Sophia Donato, Jules Jacquet, Lucas Vilour

This project compares three deep learning architectures for classifying English movie reviews from [Rotten Tomatoes](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews) as **positive** or **negative**. The three models follow the historical evolution of the field: a simple MLP, a CNN, and a Transformer with self-attention. The Transformer is also extended to predict a **star rating (0–5)**.

The full write-up is in the project report (`DL_report.pdf`).

---

## Repository structure

Each architecture lives on **its own branch**:

| Branch | Architecture | Description |
|--------|--------------|-------------|
| `simple-nn` | Fully connected network | Embedding (16 features) + 2 × (Linear + ReLU) + Dropout |
| `cnn` | Convolutional network (encoder–decoder) | Embedding + 1D convolutions / ReLU / max pooling as encoder, fully connected layers as decoder |
| `gpt` | GPT-style Transformer (bidirectional) | Pre-layer norm, multi-head self-attention, relative positional embeddings, mean pooling + linear head |

---

## Dataset

- **Source:** Rotten Tomatoes movie reviews (Kaggle), about 1.3 million reviews.
- **Split:** 90% training / 10% validation; the full test set is used for the confusion matrices.
- **Preprocessing (shared across models):** removal of special characters and extra whitespace, lowercasing, word-level tokenization, padding to a common length.
  - MLP / CNN: vocabulary limited to 10,000 words.
  - Transformer: vocabulary limited to 20,000 tokens, plus lemmatization and removal of empty reviews. Stop-word removal was tested and discarded because it hurt results (negations like "not" and "never" are stop words).
- No data augmentation was used, given the size of the dataset.

---

## Models

### MLP
- Embedding layer, two fully connected blocks with ReLU, dropout.
- Binary cross-entropy with logits (the sigmoid + BCE version was numerically unstable), Adam optimizer.

### CNN
- Embedding layer, stacked 1D convolution + ReLU + max pooling blocks, then fully connected decoder.
- Experiments on dataset size, dropout, depth and kernel size (3 / 4 / 6).
- Prone to overfitting on validation loss with long training.

### Transformer
- Adapted from G. Louppe's GPT implementation, modified for classification.
- Bidirectional attention (outperformed causal attention), relative positional encoding (outperformed absolute), pre-layer normalization (post-layer norm failed to converge).
- Learning rate scheduler comparison: StepLR, ExponentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingLR (best: **CosineAnnealingLR**).
- Early stopping with a patience of 20 epochs on validation loss.

---

## Results

### Binary sentiment classification (full test set)

| Model | Precision | Recall | F1 | Accuracy |
|-------|-----------|--------|----|----------|
| MLP | 87% | 86% | 87% | 82% |
| CNN | 84% | 92% | 88% | 82% |
| Transformer | 80% | 92% | 85% | 79% |

All three models reach roughly 80% accuracy. The MLP performs best on this task, likely because the reviews are short (about 100 words at most), so attention and positional encoding bring little advantage. The MLP and CNN plateau around 80% validation accuracy, while the Transformer is still improving and would likely benefit from longer training. The Transformer also shows a bias toward positive predictions.

### Star rating prediction (Transformer, 6 classes)

- Ratings on different scales (0–5, 0–10) were normalized to 0–5 and rounded; letter grades and unknown scales were dropped.
- Accuracy: **54.8%**, limited by the amount of usable data after filtering and by the subjectivity of ratings.

### Qualitative tests

Models were also tested on 15 new reviews (written by the us or generated with ChatGPT). See Appendices A and B of the report.

---

## Future work

- Explore the Transformer further (longer training, more compute) to improve binary classification, ideally on longer reviews.
- Improve the star prediction model with more usable data and better handling of rating scales.

---

## References

- G. Louppe, *GPT implementation* (INFO8010 course code).
- K. L. Tan et al., *RoBERTa-GRU: A hybrid deep learning model for enhanced sentiment analysis*, Applied Sciences, 2023.
- N. M. Ali et al., *Sentiment analysis for movies reviews dataset using deep learning models*, IJDKP, 2019.
- K. Dashtipour et al., *Sentiment analysis of Persian movie reviews using deep learning*, Entropy, 2021.
