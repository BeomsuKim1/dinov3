import os
from huggingface_hub import login
from transformers import pipeline
from transformers.image_utils import load_image

os.environ["HF_HOME"] = "/home/beomsu/ssd/huggingface_cache"

with open(os.path.expanduser("token.txt")) as f:
    token = f.read().strip()

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

login(token=token)

feature_extractor = pipeline(
    model="facebook/dinov3-vit7b16-pretrain-lvd1689m",
    task="image-feature-extraction", 
    device=0  # Use GPU if available
)
features = feature_extractor(image)