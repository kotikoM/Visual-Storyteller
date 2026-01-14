import os
import pickle
import re
import time
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from model import ImageCaptioningModel

IMAGE_PATH = './caption_data/Images'
CAPTIONS_PATH = './caption_data/captions.txt'
VOCAB_PATH = './trained/vocab.pkl'
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MIN_WORD_FREQ = 3  # filter words that appear less than this

START, END, PAD, UNKNOWN = '<START>', '<END>', '<PAD>', '<UNK>'


class Vocabulary:
    """Vocabulary class to handle word-to-index and index-to-word mappings"""

    def __init__(self, freq_threshold=5):
        self.itos = {0: PAD, 1: UNKNOWN, 2: START, 3: END}
        self.stoi = {PAD: 0, UNKNOWN: 1, START: 2, END: 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = text.split()
        return [self.stoi.get(token, self.stoi[UNKNOWN]) for token in tokenized_text]


class FlickrDataset(Dataset):
    """Custom Dataset for Flickr8k"""

    def __init__(self, df, image_path, vocab, transform=None):
        self.df = df
        self.image_path = image_path
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        caption = row['caption_clean']

        # Load image
        img_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Numericalize caption
        caption_vec = self.vocab.numericalize(caption)
        caption_vec = torch.tensor(caption_vec)

        return img, caption_vec


class CollateFunc:
    """Custom collate function to pad sequences in a batch"""

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return imgs, targets


def clean_caption(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\s]", "", sentence)  # punctuation
    sentence = re.sub(r"\d+", "", sentence)  # numbers
    sentence = re.sub(r"\b\w\b", "", sentence)  # single chars
    sentence = re.sub(r"\b(a|an|the)\b", "", sentence)  # articles
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return START + ' ' + sentence + ' ' + END


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for idx, (imgs, captions) in enumerate(dataloader):
        imgs = imgs.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, captions)

        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        if (idx + 1) % 100 == 0:
            print(f"  Batch [{idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, captions in dataloader:
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv(CAPTIONS_PATH)
    print(df.head())

    num_images = df["image"].nunique()
    print(f"\nNumber of unique images: {num_images}")

    all_words = df["caption"].str.split().explode()
    vocabulary_count = all_words.value_counts()

    print(f"\nNumber of unique words before filtering: {vocabulary_count.size}")
    print("Most frequently used words:", vocabulary_count.keys()[:10].tolist())

    df["caption_clean"] = df["caption"].apply(lambda x: clean_caption(x))
    all_words_after = df["caption_clean"].str.split().explode()
    vocab_after = all_words_after.value_counts()

    print(f"\nNumber of unique words after filtering: {vocab_after.size}")
    print("Most frequent after cleaning:", vocab_after.index[:10].tolist())

    df_sample = df[["caption", "caption_clean"]].sample(3)
    print("\nSample cleaned captions:")
    for idx, row in df_sample.iterrows():
        print(f"\nRow {idx}:")
        print(f"  Original: {row['caption']}")
        print(f"  Cleaned : {row['caption_clean']}")

    print("\nBuilding vocabulary...")
    vocab = Vocabulary(freq_threshold=MIN_WORD_FREQ)
    vocab.build_vocabulary(df["caption_clean"].tolist())
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved to vocab.pkl")

    unique_images = df['image'].unique()
    train_imgs, val_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)

    train_df = df[df['image'].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df['image'].isin(val_imgs)].reset_index(drop=True)

    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FlickrDataset(train_df, IMAGE_PATH, vocab, transform)
    val_dataset = FlickrDataset(val_df, IMAGE_PATH, vocab, transform)

    pad_idx = vocab.stoi[PAD]
    collate_fn = CollateFunc(pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    print("\nInitializing model...")

    model = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS, DROPOUT).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': vocab_size,
            }, 'best_model.pth')
            print(f"Best model saved with val loss: {val_loss:.4f}")

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
