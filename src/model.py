import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN Encoder using pretrained ResNet50.
    NOW returns spatial feature maps instead of single vector.
    This allows attention to focus on different image regions.
    Output: [B, num_pixels, encoder_dim] where num_pixels = 14x14 = 196
    """
    def __init__(self, encoded_image_size=14):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove last two layers (avgpool + fc)
        # Keep spatial feature maps for attention
        modules = list(resnet.children())[:-2]
        self.resnet     = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        features = self.resnet(images)              # [B, 2048, H, W]
        features = self.adaptive_pool(features)     # [B, 2048, 14, 14]
        features = features.permute(0, 2, 3, 1)    # [B, 14, 14, 2048]
        B, H, W, C = features.shape
        features = features.view(B, H*W, C)         # [B, 196, 2048]
        return features

    def fine_tune(self, fine_tune=True):
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Only fine-tune last 2 blocks
        for child in list(self.resnet.children())[6:]:
            for param in child.parameters():
                param.requires_grad = fine_tune


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) Attention for Image Captioning.
    At each decode step:
    - Computes alignment score between decoder hidden state and each image region
    - Softmax gives attention weights over 196 image pixels
    - Context = weighted sum of encoder features
    This lets the decoder FOCUS on relevant image regions when generating each word.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)   # image features → attention
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)   # hidden state → attention
        self.full_att    = nn.Linear(attention_dim, 1)             # attention score
        self.relu        = nn.ReLU()
        self.softmax     = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out    : [B, 196, encoder_dim]
        # decoder_hidden : [B, decoder_dim]
        att1    = self.encoder_att(encoder_out)              # [B, 196, attention_dim]
        att2    = self.decoder_att(decoder_hidden).unsqueeze(1)  # [B, 1, attention_dim]
        att     = self.full_att(self.relu(att1 + att2)).squeeze(2)  # [B, 196]
        alpha   = self.softmax(att)                          # [B, 196] attention weights
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [B, encoder_dim]
        return context, alpha


class DecoderWithAttention(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention.
    At each step:
    1. Attention computes which image regions to focus on
    2. Context vector = weighted image features
    3. LSTM takes [word_embedding + context] as input
    4. Predicts next word
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim,
                 vocab_size, encoder_dim=2048, dropout=0.3):
        super().__init__()
        self.encoder_dim  = encoder_dim
        self.decoder_dim  = decoder_dim
        self.attention    = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding    = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout      = nn.Dropout(dropout)
        self.lstm         = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h       = nn.Linear(encoder_dim, decoder_dim)
        self.init_c       = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta       = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid      = nn.Sigmoid()
        self.fc           = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_enc = encoder_out.mean(dim=1)              # [B, encoder_dim]
        h = self.init_h(mean_enc)                       # [B, decoder_dim]
        c = self.init_c(mean_enc)                       # [B, decoder_dim]
        return h, c

    def forward(self, encoder_out, captions):
        B          = encoder_out.size(0)
        vocab_size = self.fc.out_features
        embeddings = self.dropout(self.embedding(captions[:, :-1]))  # [B, T-1, E]
        T          = embeddings.size(1)
        h, c       = self.init_hidden_state(encoder_out)
        predictions = torch.zeros(B, T, vocab_size).to(encoder_out.device)
        alphas      = torch.zeros(B, T, encoder_out.size(1)).to(encoder_out.device)

        for t in range(T):
            context, alpha = self.attention(encoder_out, h)
            gate    = self.sigmoid(self.f_beta(h))
            context = gate * context
            lstm_in = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c    = self.lstm(lstm_in, (h, c))
            pred    = self.fc(self.dropout(h))
            predictions[:, t, :] = pred
            alphas[:, t, :]      = alpha

        return predictions, alphas

    def generate(self, encoder_out, vocab, max_len=30, device="cpu"):
        B          = encoder_out.size(0)
        h, c       = self.init_hidden_state(encoder_out)
        sos_id     = vocab["<sos>"]
        eos_id     = vocab["<eos>"]
        input_tok  = torch.tensor([sos_id]).to(device)
        result     = []
        inv_vocab  = {v: k for k, v in vocab.items()}

        for _ in range(max_len):
            emb            = self.dropout(self.embedding(input_tok))  # [1, E]
            context, alpha = self.attention(encoder_out, h)
            gate           = self.sigmoid(self.f_beta(h))
            context        = gate * context
            lstm_in        = torch.cat([emb, context], dim=1)
            h, c           = self.lstm(lstm_in, (h, c))
            pred           = self.fc(self.dropout(h))
            top            = pred.argmax(1).item()
            if top == eos_id: break
            if top not in (vocab["<pad>"], vocab["<sos>"]):
                result.append(inv_vocab.get(top, ""))
            input_tok = torch.tensor([top]).to(device)

        return " ".join(result)


class ImageCaptioningWithAttention(nn.Module):
    """
    Full Image Captioning model with Attention.
    Encoder: ResNet50 → spatial feature maps [B, 196, 2048]
    Attention: Bahdanau → focus on relevant regions
    Decoder: LSTM + attention context → word predictions
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim,
                 vocab_size, encoder_dim=2048, dropout=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderWithAttention(
            attention_dim, embed_dim, decoder_dim,
            vocab_size, encoder_dim, dropout
        )

    def forward(self, images, captions):
        encoder_out          = self.encoder(images)
        predictions, alphas  = self.decoder(encoder_out, captions)
        return predictions, alphas