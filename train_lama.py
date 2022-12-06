import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import math
import torch

from lama import *
from utils import setup_seed
from dataset import build_dataset
from tqdm import tqdm
from torchvision.utils import save_image
from focal_frequency_loss import FocalFrequencyLoss
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='../data')
    parser.add_argument('--block_size', type=int, default=128)
    args = parser.parse_args()

    setup_seed(args.seed)

    train_dataloader, val_dataloader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LaMa(mask_ratio=args.mask_ratio)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate)
    l1 = nn.L1Loss().to(device)
    ffl = FocalFrequencyLoss().to(device)
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        losses = []
        for img in train_dataloader:
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            masked_img = img * mask
            inv_mask = torch.ones_like(mask).to(device) - mask # marks 1 to masked area
            loss = l1(predicted_img, img) + ffl(predicted_img, img)
            mse = torch.mean((predicted_img - img) ** 2 * inv_mask) / args.mask_ratio
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(mse.item())
            loss_avg = sum(losses) / len(losses)
            psnr_avg = 10 * math.log10(1 / loss_avg)
            if step_count % 10 == 0:
                print(f"step: {step_count} \t|  PSNR loss: "+ str(psnr_avg) + "dB"
                      f"\t|  Train loss: {loss_avg}")
                save_image([img[0], masked_img[0], predicted_img[0]], "result.png")
        print(f'In epoch {e}, average training loss is {psnr_avg} dB.')

        # ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_losses = []
            for val_img in val_dataloader:
                val_img = val_img.to(device)
                predicted_val_img, mask = model(val_img)
                predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
                val_loss = torch.mean((predicted_val_img - val_img) ** 2 * mask) / args.mask_ratio
                val_losses.append(val_loss.item())
                val_mse_avg = sum(val_losses) / len(val_losses)
                val_psnr_avg = 10 * math.log10(1 / val_mse_avg)
        print(f'In epoch {e}, average validation loss is {val_psnr_avg} dB.')
        ''' save model '''
        if e % 10 == 0:
            torch.save(model.state_dict(), f"lama-{e}.pth")