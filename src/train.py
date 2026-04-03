import torch
import torch.nn as nn
import pickle
import os
import random
import time
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from model_attention import ImageCaptioningWithAttention
from preprocess import encode_caption, train_transform, val_transform

# ══════════════════════════════════════════════════════════════════
# HYPERPARAMETERS — Attention Model
# ══════════════════════════════════════════════════════════════════
ATTENTION_DIM = 256   # Attention layer dimension
EMBED_DIM     = 256   # Word embedding dimension
DECODER_DIM   = 512   # LSTM hidden state size
ENCODER_DIM   = 2048  # ResNet50 output channels
DROPOUT       = 0.3
BATCH_SIZE    = 32
N_EPOCHS      = 10
LR            = 4e-4
CLIP          = 5.0
MAX_LEN       = 30
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── TRAINING RESULTS (filled after training) ──────────────────────
# Epoch 01 | Train: x.xxxx | Val: x.xxxx
# Epoch 02 | Train: x.xxxx | Val: x.xxxx
# ...
# Best Val Loss: x.xxxx
# Compare with simple model best val: 2.4957
# ──────────────────────────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    def __init__(self, image_captions, vocab, img_dir, transform):
        self.items     = []
        self.img_dir   = img_dir
        self.transform = transform
        for img_name, captions in image_captions.items():
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path): continue
            for cap in captions:
                self.items.append((img_name, encode_caption(cap, vocab, MAX_LEN)))
        random.shuffle(self.items)
        print(f"Dataset size: {len(self.items)}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img_name, caption = self.items[i]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        return self.transform(img), torch.tensor(caption)

def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs, 0)
    caps = pad_sequence(caps, batch_first=True, padding_value=0)
    return imgs, caps

if __name__ == "__main__":
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Using device: {DEVICE}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(data_dir, "vocab.pkl"),          "rb") as f: vocab          = pickle.load(f)
    with open(os.path.join(data_dir, "image_captions.pkl"), "rb") as f: image_captions = pickle.load(f)

    # Find images
    img_dir = None
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            if "image" in d.lower() or "img" in d.lower():
                img_dir = os.path.join(root, d)
                break
        if img_dir: break
    assert img_dir, "Images folder not found!"
    print(f"Images: {img_dir} | Vocab: {len(vocab)}")

    all_imgs   = list(image_captions.keys())
    random.shuffle(all_imgs)
    split      = int(0.9 * len(all_imgs))
    train_imgs = {k: image_captions[k] for k in all_imgs[:split]}
    val_imgs   = {k: image_captions[k] for k in all_imgs[split:]}

    train_dl = DataLoader(
        Flickr8kDataset(train_imgs, vocab, img_dir, train_transform),
        BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dl = DataLoader(
        Flickr8kDataset(val_imgs, vocab, img_dir, val_transform),
        BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = ImageCaptioningWithAttention(
        ATTENTION_DIM, EMBED_DIM, DECODER_DIM,
        len(vocab), ENCODER_DIM, DROPOUT
    ).to(DEVICE)

    # Only train decoder + attention (encoder frozen except last 2 blocks)
    optimizer = torch.optim.Adam([
        {"params": model.decoder.parameters(), "lr": LR},
        {"params": model.encoder.resnet.parameters(), "lr": LR/10}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    best_val  = float("inf")
    ckpt_path = os.path.join(ckpt_dir, "best_attention_model.pt")

    for epoch in range(1, N_EPOCHS+1):
        # Train
        model.train()
        tr_loss = 0
        t0 = time.time()
        for imgs, caps in train_dl:
            imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
            optimizer.zero_grad()
            preds, alphas = model(imgs, caps)
            loss = criterion(
                preds.reshape(-1, len(vocab)),
                caps[:, 1:].reshape(-1) if preds.size(1) == caps.size(1)-1
                else caps.reshape(-1)
            )
            # Doubly stochastic attention regularization
            loss += 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, caps in val_dl:
                imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
                preds, _   = model(imgs, caps)
                val_loss  += criterion(
                    preds.reshape(-1, len(vocab)),
                    caps[:, 1:].reshape(-1) if preds.size(1) == caps.size(1)-1
                    else caps.reshape(-1)
                ).item()
        val_loss /= len(val_dl)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{N_EPOCHS} | Train: {tr_loss:.4f} | Val: {val_loss:.4f} | Time: {time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state"  : model.state_dict(),
                "vocab"        : vocab,
                "attention_dim": ATTENTION_DIM,
                "embed_dim"    : EMBED_DIM,
                "decoder_dim"  : DECODER_DIM,
                "encoder_dim"  : ENCODER_DIM,
                "dropout"      : DROPOUT,
            }, ckpt_path)
            print(f"  ✓ Saved (val: {best_val:.4f})")

    print(f"\nTraining complete!")
    print(f"Simple model best val  : 2.4957")
    print(f"Attention model best val: {best_val:.4f}")