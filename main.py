import os
import numpy as np
from huggingface_hub import login
from transformers import pipeline
from transformers.image_utils import load_image

os.environ["HF_HOME"] = "/home/beomsu/ssd/huggingface_cache"

with open(os.path.expanduser("token.txt")) as f:
    token = f.read().strip()

path = "/home/beomsu/Downloads/iStock-1052880600.jpg"
image = load_image(path)

login(token=token)

feature_extractor = pipeline(
    model="facebook/dinov3-vith16plus-pretrain-lvd1689m",
    task="image-feature-extraction", 
    device=0  # Use CPU; change to 0 for GPU if available
)
features = np.array(feature_extractor(image))
print(features.shape)