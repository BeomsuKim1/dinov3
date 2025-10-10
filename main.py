import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
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

plt.figure()
plt.imshow(image)
plt.axis('off')

device = "cuda" # cpu or cuda
print("Device:", device)

MODEL_DINOV3 = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(MODEL_DINOV3)
model = AutoModel.from_pretrained(MODEL_DINOV3).to(device)
patch_size = model.config.patch_size
print("Patch size:", patch_size)
print("Num register tokens:", model.config.num_register_tokens)

inputs = processor(images=image, 
                   return_tensors="pt", 
                   device=device, 
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
print("patch_features_flat.shape:", patch_features_flat.shape)
patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))
print("patch_features.shape:", patch_features.shape)

B, H, W, C = patch_features.shape
N = H * W

X = np.asarray(patch_features_flat.reshape(B*N, C).cpu())
pca = PCA(n_components=3, whiten=True)
pca.fit(X)
print("X.shape:", X.shape)

X_pca_flat = pca.transform(X)
print("X_pca_flat.shape:", X_pca_flat.shape)

X_pca = X_pca_flat.reshape(B, H, W, 3)
print("X_pca.shape:", X_pca.shape)

projected_image = 1 / (1 + np.exp(-2 * X_pca))

for i in range(B):
  plt.figure()
  plt.imshow(projected_image[i])
  plt.axis('off')

eps = 1e-12
row_norms = np.linalg.norm(X, axis=1, keepdims=True)
row_norms = np.maximum(row_norms, eps)
Xn = X / row_norms

D = np.zeros((H, W), dtype=np.float32)

fig, ax = plt.subplots()
im = ax.imshow(D, cmap="viridis_r", origin="upper", interpolation="nearest", vmin=0.0, vmax=2.0)
marker, = ax.plot([], [], "r+", markersize=20, markeredgewidth=4)  # click marker
plt.tight_layout()
plt.axis('off')

def on_click(event):
    if event.inaxes is not ax or event.xdata is None or event.ydata is None:
        return
    j = int(round(event.xdata))
    i = int(round(event.ydata))
    i = max(0, min(H-1, i))
    j = max(0, min(W-1, j))
    idx = i * W + j

    x0 = Xn[idx]

    sim = Xn @ x0
    dist = 1.0 - sim
    D = dist.reshape(H, W)

    im.set_data(D)
    im.set_clim(0.0, 2.0)
    marker.set_data([j], [i])
    fig.canvas.draw_idle()

cid = fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()