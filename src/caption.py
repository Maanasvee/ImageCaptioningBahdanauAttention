import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from model import ImageCaptioningWithAttention
from preprocess import val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_attention_model(checkpoint_path):
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    vocab = ckpt["vocab"]
    model = ImageCaptioningWithAttention(
        ckpt["attention_dim"], ckpt["embed_dim"],
        ckpt["decoder_dim"],   len(vocab),
        ckpt["encoder_dim"],   ckpt["dropout"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab

def generate_caption_attention(image_path, model, vocab):
    img     = Image.open(image_path).convert("RGB")
    img     = val_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        encoder_out = model.encoder(img)
        caption     = model.decoder.generate(encoder_out, vocab, device=DEVICE)
    return caption

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt     = os.path.join(BASE_DIR, "checkpoints", "best_attention_model.pt")
    model, vocab = load_attention_model(ckpt)
    print("Attention model loaded!")