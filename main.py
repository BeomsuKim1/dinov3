import torch

REPO_DIR = "/home/beomsu/dinov3/dinov3"

dinov3_vith16plus = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights="/home/beomsu/ssd/dinov3_weights/dinov3_vith16plus_pretrain_lvd1689m")
#dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
