import os
import torch
from huggingface_hub import login
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

os.environ["HF_HOME"] = "/home/beomsu/ssd/huggingface_cache"

with open(os.path.expanduser("token.txt")) as f:
    token = f.read().strip()
login(token=token)

path = "/home/beomsu/Downloads/iStock-1052880600.jpg"
image = load_image(path)
print("Image size:", image.height, image.width)

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")
model = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")
patch_size = model.config.patch_size
print("Patch size:", patch_size)
print("Num register tokens:", model.config.num_register_tokens)

inputs = processor(images=image, 
                   return_tensors="pt", 
                   do_resize=False, 
                   do_center_crop=False)
print("Preprocessed image size:", inputs.pixel_values.shape)

batch_size, _, img_height, img_width = inputs.pixel_values.shape
num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
num_patches_flat = num_patches_height * num_patches_width

with torch.inference_mode():
  outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
assert last_hidden_states.shape == (batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)

cls_token = last_hidden_states[:, 0, :]
patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))