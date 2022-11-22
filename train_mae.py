import os
import argparse
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import *
from utils import setup_seed
from dataset import build_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--dataset', type=str, default='/local_datasets/nnvc')
    parser.add_argument('--block_size', type=int, default=128)
    args = parser.parse_args()

    setup_seed(args.seed)

    train_dataloader, val_dataloader = build_dataset(args)
    writer = SummaryWriter(os.path.join('logs', 'vimeo90k', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio)
    #model.load_state_dict(torch.load(args.model_path).state_dict())
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        losses = []
        for img in train_dataloader:
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            mse_avg = sum(losses) / len(losses)
            psnr_avg = 10 * math.log10(1 / mse_avg)
            if step_count % 100 == 0:
                print(f"step: {step_count} \t|  PSNR loss: "+ str(psnr_avg) + "dB"
                      f"\t|  MSE loss: {mse_avg}")
        lr_scheduler.step()
        writer.add_scalar('train loss', psnr_avg, global_step=e)
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
        writer.add_scalar('mae_loss', val_psnr_avg, global_step=e)
        ''' save model '''
        if e % 10 == 0:
            torch.save(model.state_dict(), f"vit-t-mae-{e}.pth")