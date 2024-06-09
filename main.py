import wandb
import os
import sys
import time
import random
import copy
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
from easydict import EasyDict
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from AnoViT.model import Decoder_r
from AnoViT.funcs import EarlyStop
from AnoViT.dataloader import get_dataloader
from AnoViT.utils import (time_string, convert_secs2time, AverageMeter,
                          print_log)
from AnoViT.timm.models.vision_transformer import *
from AnoViT.timm import create_model
from AnoViT.funcs import plt_show, denormalization
from AnoViT.mean_std import obj_stats_384
def train(args, scaler, model, epoch, train_loader, optimizer, log):
    model.train()
    MSE = nn.MSELoss()

    for (x, _, _) in tqdm(train_loader):
        x = x.to(args.device)
        optimizer.zero_grad()
        if args.amp:
            with amp.autocast():
                x_hat = model(x)
                loss = MSE(x, x_hat)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    print_log('Train Epoch: {} | MSE Loss: {:.6f}'.format(epoch, loss),
              log)


def val(args, model, epoch, val_loader, log):
    model.eval()
    MSE = nn.MSELoss()

    for (x, _, _) in tqdm(val_loader):
        x = x.to(args.device)
        with torch.no_grad():
            x_hat = model(x)
            loss = MSE(x, x_hat)
            
            if epoch % 10 == 0:
                plt_show(args, x_hat, x, epoch)

    print_log(('Valid Epoch: {} | MSE Loss: {:.6f}'.format(epoch, loss)),
              log)

    return loss, model


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def save_test_image(test_loader, test_imgs, recon_imgs, mean, std, seg_scores,
                    gt_mask_list):
    for num in range(len(test_loader)):
        # visualize the score map
        if num in [13, 62, 67, 71, 72, 73, 74, 75, 76, 77]:
            if test_imgs[num].dtype != "uint8":
                test_imgs[num] = denormalization(test_imgs[num], mean, std)

            if recon_imgs[num].dtype != "uint8":
                recon_imgs[num] = denormalization(recon_imgs[num], mean, std)

            scores_img = seg_scores[num]

            fig, plots = plt.subplots(1, 4)

            fig.set_figwidth(9)
            fig.set_tight_layout(True)
            plots = plots.reshape(-1)
            plots[0].imshow(test_imgs[num])
            plots[1].imshow(recon_imgs[num])
            plots[2].imshow(scores_img, cmap='jet', alpha=0.35)
            plots[3].imshow(gt_mask_list[num], cmap=plt.cm.gray)

            plots[0].set_title("real")
            plots[1].set_title("reconst")
            plots[2].set_title("anomaly score")
            plots[3].set_title("gt mask")

            plt.savefig('result/test_image/{}_{}_{}.png'.format(
                args.model, args.obj, num))


def test(model, test_loader, device=torch.device('cuda:0')):
    model.eval()
    MSE = nn.MSELoss(reduction='none')
    det_scores, seg_scores = [],[]
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []

    det_sig, seg_sig = 15,6
    for (x, label, mask) in tqdm(test_loader):
        mask = mask.squeeze(0)
        test_imgs.extend(x.cpu().numpy())
        gt_list.extend(label.cpu().numpy())
        gt_mask_list.extend(mask.cpu().numpy())
        score = 0
        with torch.no_grad():
            x = x.to(device)
            x_hat = model(x)

            mse = MSE(x,x_hat)
            score = mse

        score = score.cpu().numpy()
        score = score.mean(1) #  channel간 평균

        det_score, seg_score = copy.deepcopy(score), copy.deepcopy(score)

        for i in range(det_score.shape[0]):
            det_score[i] = gaussian_filter(det_score[i], sigma=det_sig)
        det_scores.extend(det_score)

        for i in range(seg_score.shape[0]):
            seg_score[i] = gaussian_filter(seg_score[i], sigma=seg_sig)
        seg_scores.extend(seg_score)

        recon_imgs.extend(x_hat.cpu().numpy())
    return det_scores, seg_scores, test_imgs, recon_imgs, gt_list, gt_mask_list


if __name__ == "__main__":
    #check torch version & device
    print ("Python version:[%s]."%sys.version)
    print ("PyTorch version:[%s]."%torch.__version__)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print ("device:[%s]."%device)
    run_id = wandb.util.generate_id()
    seed = 42
    set_seed(seed)
    args = EasyDict({
        'n_gpu': 1,
        'image_size': 384,
        'patch_size': 16,
        'device': 'cuda',
        'batch_size': 2,
        'num_workers': 16,
        'epochs': 20,
        'lr': 2e-4,
        'wd': 1e-5,
        'obj': 'grid',
        'val_ratio': 0.4,
        'save_dir': '/media/mostafahaggag/Shared_Drive/selfdevelopment/LG_ES_anomaly/AnoViT/result',
        'dataset_path': '/media/mostafahaggag/Shared_Drive/selfdevelopment/LG_ES_anomaly/data/mvtec/mvtec/',
        'model': 'vit',
        'amp': True,
        'seed': 42,
        'beta1': 0.5,
        'beta2': 0.999
    })
    device = args.device
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open('model_training_log_{}_{}.txt'.format(args.obj, args.model), 'w')
    scaler = amp.GradScaler()
    '''
    initializes an instance of the GradScaler class from PyTorch's torch.cuda.amp (Automatic Mixed Precision) module. 
    The GradScaler is used to facilitate mixed precision training, 
    which can help improve performance by using both 16-bit (half precision) and 32-bit (single precision) 
    floating-point operations during training.
    '''
    model = create_model('vit_base_patch16_384', pretrained=True)
    decmodel = nn.Sequential(model, Decoder_r(args))
    decmodel.to(device)
    train_loader, val_loader, _ = get_dataloader(args)
    optimizer = torch.optim.Adam(
        params=decmodel.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2))
    save_name = os.path.join(args.save_dir, 'model/{}_{}_best_model.pt'.format(args.obj, args.model))
    early_stop = EarlyStop(patience=12, save_name=save_name)
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(1, args.epochs + 1):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)

        train(args, scaler, decmodel, epoch, train_loader, optimizer, log)
        val_loss, save_model = val(args, decmodel, epoch, val_loader, log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()
    torch.save(save_model.state_dict(), 'AnoViT.pt')
    decmodel = nn.Sequential(model, Decoder_r(args))
    decmodel.load_state_dict(torch.load('AnoViT.pt'), strict=False)
    decmodel.to(device)
    train_loader, valid_loader, test_loader = get_dataloader(args)
    
    ##
    det_scores, seg_scores, test_imgs, recon_imgs, gt_list, gt_mask_list = test(decmodel, test_loader, device)

    # det_scores = np.asarray(det_scores)
    # max_anomaly_score = det_scores.max()
    # min_anomaly_score = det_scores.min()
    # det_scores = (det_scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    seg_scores = np.asarray(seg_scores)
    max_anomaly_score = seg_scores.max()
    min_anomaly_score = seg_scores.min()
    seg_scores = (seg_scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    gt_mask = np.asarray(gt_mask_list)
    gt_mask = gt_mask.astype('int')
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), seg_scores.flatten())
    print('pixel ROCAUC: %.2f' % (per_pixel_rocauc))