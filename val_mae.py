import os
import argparse
import math
import torch

from tqdm import tqdm

from model import *
from utils import setup_seed
from dataset import build_test_dataset

from torchvision.utils import save_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--dataset', type=str, default='../data')
    parser.add_argument('--block_size', type=int, default=128)
    args = parser.parse_args()

    setup_seed(args.seed)

    val_dataloader = build_test_dataset(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio)
    model.load_state_dict(torch.load(args.model_path).state_dict())
    model.to(device)

    model.eval()
    with torch.no_grad():
        val_losses = []
        i = 0
        for val_img in tqdm(iter(val_dataloader)):
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            val_loss = torch.mean((predicted_val_img - val_img) ** 2 * mask) / args.mask_ratio
            val_losses.append(val_loss.item())
            val_mse_avg = sum(val_losses) / len(val_losses)
            val_psnr_avg = 10 * math.log10(1 / val_mse_avg)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=1, v=3)
            save_image(img, f"./recon_{i + 1}.png")
            i = i + 1
        print(f"PSNR : {val_psnr_avg}dB")