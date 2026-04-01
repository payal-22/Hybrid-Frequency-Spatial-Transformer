import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
from losses.loss_functions import *

from losses.losses import *
from torchvision.models import vgg16
from kornia.losses import SSIMLoss



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Main-m', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='UIEB', type=str, help='dataset name')
parser.add_argument('--exp', default='data', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1', type=str, help='GPUs used for training')
parser.add_argument('--pretrained_model', default='saved_models/indoor0/', type=str, help='pretrained_model path')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# def load_model(backbone, model_dir):
# 	if backbone == 'dehazeformer-s1':
# 		net = dehazeformer_s()
# 		# net.to(device)
# 		# net = nn.DataParallel(net, device_ids=device_ids)
# 		model_path = os.path.join(model_dir, 'dehazeformer-s1.pth')
# 		net.load_state_dict(torch.load(model_path))
#
# 	return net


def train(train_loader, network, criterion, optimizer, scaler):
    Losses = AverageMeter()

    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.cuda()
    torch.cuda.empty_cache()

    network.train()

    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        prior_img = batch['prior'].cuda()
        with autocast(args.no_autocast):
            #output, T = network(source_img, prior_img)
            I_rec, T = network(source_img, prior_img)
            # loss = criterion(output, target_img)
            ldc = dark_channel(source_img, T)
            lbc = bright_channel(source_img, T)
            stru = MyLoss()
            lssm = SSIMLoss(11)
            ls = stru(I_rec, target_img)
            pre = PerpetualLoss(vgg_model)
            lp = pre(I_rec, target_img)
            lss = lssm(I_rec,target_img)
            Loss = ls * 1+ ldc * 2e-3 + lbc * 3e-2 + 0.3 * lp + 1 * lss
            # Loss = ls * 1+ 1 * lss

        Losses.update(Loss.item())

        optimizer.zero_grad()
        scaler.scale(Loss).backward()
        scaler.step(optimizer)
        scaler.update()


    return Losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        prior_img = batch['prior'].cuda()
        with torch.no_grad():
            I_rec, T = network(source_img, prior_img)
            I_rec = I_rec.clamp_(-1, 1)
        mse_loss = F.mse_loss(I_rec * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    setting_filename = os.path.join(base_dir, 'configs', args.exp, args.model+'.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join(base_dir, 'configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()
    #model_path = os.path.join(args.pretrained_model, 'dehazeformer-s1.pth')
    #checkpoint = torch.load(model_path)
    #network.load_state_dict(checkpoint['state_dict'])

    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                                setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                              setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
        print('==> Start training, current model name: ' + args.model)
        # print(network)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        best_psnr = 0
        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion,optimizer, scaler)
            # print('loss			',loss)
            writer.add_scalar('train_loss', loss, epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr = valid(val_loader, network)
                # print('avg_psnr'	,avg_psnr)
                writer.add_scalar('valid_psnr', avg_psnr, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model+'.pth'))

                writer.add_scalar('best_psnr', best_psnr, epoch)

    else:
        print('==> Existing trained model')
        exit(1)
