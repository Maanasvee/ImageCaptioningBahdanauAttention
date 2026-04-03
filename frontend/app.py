from flask import Flask, request, jsonify, render_template
import sys, os, uuid, io
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from caption import load_attention_model, generate_caption_attention

app    = Flask(__name__)
CKPT   = os.path.join(os.path.dirname(__file__), '../checkpoints/best_attention_model.pt')
UPLOAD = os.path.join(os.path.dirname(__file__), 'static/uploads')
os.makedirs(UPLOAD, exist_ok=True)

model, vocab = load_attention_model(CKPT)
print("Attention model loaded!")

@app.route("/")
def index(): return render_template("index.html")

@app.route("/caption", methods=["POST"])
def caption():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})
    file     = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD, filename)
    file.stream.seek(0)
    img = Image.open(io.BytesIO(file.stream.read())).convert("RGB")
    img.save(filepath)
    cap = generate_caption_attention(filepath, model, vocab)
    os.remove(filepath)
    return jsonify({"caption": cap})

if __name__ == "__main__":
    print("Starting at http://localhost:5002")
    app.run(debug=True, port=5002)