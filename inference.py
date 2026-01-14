import pickle
from collections import Counter

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import ImageCaptioningModel

# Configuration
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
MODEL_PATH = './trained/best_model.pth'
VOCAB_PATH = './trained/vocab.pkl'

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


def load_model_and_vocab(device):
    """Load trained model and vocabulary"""
    # Load vocabulary
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)

    # Load model
    model = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, vocab_size, NUM_LAYERS)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, vocab


def generate_caption(image_path, model, vocab, device, max_length=20):
    """Generate caption for a given image"""
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        caption_ids = model.generate_caption(image_tensor, vocab, max_length)

    # Convert IDs to words
    caption_words = [vocab.itos[idx] for idx in caption_ids]

    # Remove startseq and endseq tokens
    caption_words = [word for word in caption_words if word not in [START, END]]

    caption = ' '.join(caption_words)

    return caption, image


def visualize_prediction(image_path, caption):
    """Display image with generated caption"""
    image = Image.open(image_path)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated Caption:\n{caption}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


def batch_inference(image_paths, model, vocab, device):
    """Generate captions for multiple images"""
    results = []

    for img_path in image_paths:
        try:
            caption, _ = generate_caption(img_path, model, vocab, device)
            results.append({
                'image_path': img_path,
                'caption': caption
            })
            print(f"\nImage: {img_path}")
            print(f"Caption: {caption}")
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            results.append({
                'image_path': img_path,
                'caption': None,
                'error': str(e)
            })

    return results


def visualize_grid(results, grid_size=(2, 3)):
    """Display multiple images with captions in a grid"""
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()

    for idx, result in enumerate(results):
        if idx >= len(axes):
            break

        img_path = result['image_path']
        caption = result.get('caption', 'Error generating caption')

        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')

            # Wrap caption text for better display
            wrapped_caption = '\n'.join([caption[i:i + 50] for i in range(0, len(caption), 50)])
            axes[idx].set_title(wrapped_caption, fontsize=10, pad=10)
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error loading image', ha='center', va='center')
            axes[idx].axis('off')

    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def get_random_images(image_dir, num_images=6):
    """Get random images from the directory"""
    import os
    import random

    # Get all image files
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Randomly sample
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    # Create full paths
    image_paths = [os.path.join(image_dir, img) for img in selected_images]

    return image_paths


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and vocabulary
    print("Loading model and vocabulary...")
    model, vocab = load_model_and_vocab(device)
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {len(vocab)}")

    print("BATCH INFERENCE WITH GRID VISUALIZATION")
    print("=" * 60)

    try:
        # Get 6 random images from the dataset
        image_dir = './caption_data/Images'
        print(f"\nSelecting 6 random images from {image_dir}...")
        random_image_paths = get_random_images(image_dir, num_images=6)

        print(f"Selected images:")
        for path in random_image_paths:
            print(f"  - {path}")

        # Generate captions for all images
        print("\nGenerating captions...")
        results = batch_inference(random_image_paths, model, vocab, device)

        # Display in grid
        print("\nDisplaying results in grid...")
        visualize_grid(results, grid_size=(2, 3))

    except FileNotFoundError:
        print(f"Error: Image directory not found at {image_dir}")
        print("Please update the image_dir variable with the correct path.")
    except Exception as e:
        print(f"Error during batch inference: {e}")


if __name__ == '__main__':
    main()
