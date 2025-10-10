import os
import torch

os.environ["TORCH_HOME"] = "/home/beomsu/ssd/torch_cache"
REPO_DIR = "/home/beomsu/dinov3/dinov3"
WEIGHTS_DIR = "/home/beomsu/ssd/dinov3_weights/"

#dinov3_vith16plus = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=WEIGHTS_DIR + 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=WEIGHTS_DIR + 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth')
