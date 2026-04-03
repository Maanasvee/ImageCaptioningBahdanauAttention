"""
Comparison: Simple Encoder-Decoder vs Encoder-Decoder with Attention
Dataset: Flickr8k
"""
import torch
import os
import sys
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from caption import load_model, generate_caption
from caption_attention import load_attention_model, generate_caption_attention
from preprocess import val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════
# MODEL COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════
#
# Simple Encoder-Decoder:
#   Encoder : ResNet50 → single 256-dim vector
#   Decoder : LSTM takes image feature as first input
#   Attention: NONE
#   Best Val Loss: 2.4957
#
# Encoder-Decoder with Bahdanau Attention:
#   Encoder : ResNet50 → spatial maps [196, 2048]
#   Attention: score = v·tanh(W1·encoder + W2·hidden)
#              context = weighted sum of 196 spatial features
#   Decoder : LSTM takes [word + context] at each step
#   Best Val Loss: (filled after training)
#
# Key Difference:
#   Simple model sees the WHOLE image once as a fixed vector.
#   Attention model can FOCUS on different image regions
#   for each word it generates — like a human looking at
#   different parts of an image while describing it.
# ══════════════════════════════════════════════════════════════════

def compare_models(image_path):
    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_simple = os.path.join(BASE_DIR, "checkpoints", "best_caption_model.pt")
    ckpt_attn   = os.path.join(BASE_DIR, "checkpoints", "best_attention_model.pt")

    print(f"\nImage: {image_path}")
    print("="*60)

    # Simple model
    if os.path.exists(ckpt_simple):
        model_s, vocab_s = load_model(ckpt_simple)
        cap_simple = generate_caption(image_path, model_s, vocab_s)
        print(f"Simple Model    : {cap_simple}")
    else:
        print("Simple model checkpoint not found!")

    # Attention model
    if os.path.exists(ckpt_attn):
        model_a, vocab_a = load_attention_model(ckpt_attn)
        cap_attn = generate_caption_attention(image_path, model_a, vocab_a)
        print(f"Attention Model : {cap_attn}")
    else:
        print("Attention model checkpoint not found — train first!")

    print("="*60)
    print("\nPerformance Comparison:")
    print(f"  Simple Model  Best Val Loss : 2.4957")
    print(f"  Attention Model Best Val Loss: (check after training)")
    print(f"\nWhy Attention is better:")
    print(f"  - Focuses on relevant image regions per word")
    print(f"  - Generates more descriptive captions")
    print(f"  - Lower validation loss = better generalization")

if __name__ == "__main__":
    # Test with a sample image
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    if test_img and os.path.exists(test_img):
        compare_models(test_img)
    else:
        print("Usage: python compare.py path/to/image.jpg")