import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    Lightweight CNN encoder using pre-trained ResNet18
    """

    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use ResNet18 for lightweight model
        resnet = models.resnet18(pretrained=True)

        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Freeze ResNet parameters (optional - for faster training)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Linear layer to project to embedding size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    """
    LSTM-based decoder for caption generation
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to predict next word
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        """
        features: (batch_size, embed_size) - image features from encoder
        captions: (batch_size, seq_length) - tokenized captions
        """
        # Embed captions (excluding last word for teacher forcing)
        embeddings = self.embed(captions[:, :-1])

        # Concatenate image features with caption embeddings
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings), dim=1)

        # Pass through LSTM
        hiddens, _ = self.lstm(embeddings)

        # Apply dropout and linear layer
        outputs = self.linear(self.dropout(hiddens))

        return outputs

    def generate_caption(self, features, vocab, max_length=20):
        """
        Generate caption for a single image using greedy search
        """
        result = []
        states = None

        inputs = features.unsqueeze(1)

        for i in range(max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)

            result.append(predicted.item())

            inputs = self.embed(predicted).unsqueeze(1)

        return result


class ImageCaptioningModel(nn.Module):
    """
    Complete image captioning model combining encoder and decoder
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, dropout)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_length=20):
        features = self.encoder(image)
        caption_ids = self.decoder.generate_caption(features, vocab, max_length)
        return caption_ids
