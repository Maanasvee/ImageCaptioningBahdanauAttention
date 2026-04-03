import kagglehub
import shutil
import os

# Download Flickr8k dataset
path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)

os.makedirs("../data", exist_ok=True)

# Copy all files to local data folder
for item in os.listdir(path):
    src  = os.path.join(path, item)
    dst  = os.path.join("../data", item)
    if os.path.isdir(src):
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
    else:
        shutil.copy(src, dst)
    print(f"Copied: {item}")

print("Done! Check data/ folder.")