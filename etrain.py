import torch
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
from models.RADAR import RADAR
from utils.tdataloader import get_loader,test_dataset
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from valid_f1 import f1_mae_torch
import logging

file = open("log/RADAR.txt", "a")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True



def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, test_loader,model, optimizer, epoch,writer):
    model.train()
    last_f1 = 0
    loss_record3, loss_record2, loss_record1, loss_recorde = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()
        # ---- forward ----
        pre,sem,edge,_ = model(images)
        # ---- loss function ----
        loss1 = model.compute_loss(preds=pre,targets=gts)
        print('loss:',loss1[1])
        sem_loss = model.compute_loss(preds=sem,targets=gts)
        print('semloss:', sem_loss[1])
        loss2 = model.compute_boundaryloss(edge[0],edges)
        print('edge_loss:', loss2)
        loss=loss1[1]+0.4*sem_loss[1]+20*loss2

        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record1.update(loss1[1].data, opt.batchsize)
        loss_recorde.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))
            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]\n'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)

    last_f1 = [0 for x in range(1)] #dateset num
    if (epoch + 1) % 1 == 0 or (epoch + 1) == opt.epoch:
        logging.info("Start test...")
        tmp_f1,tmp_mae = val(test_loader,model,epoch,writer)
        for fi in range(len(last_f1)):
            if (tmp_f1[fi] > last_f1[fi]):
                tmp_out = 1
            if (tmp_out):
                last_f1 = tmp_f1
                print('last_f1',last_f1)
                torch.save(model.state_dict(), save_path + 'RADAR-%d.pth' % epoch)
                print('[Saving Snapshot:]', save_path + 'RADAR-%d.pth' % epoch)
                file.write('[Saving Snapshot:]' + save_path + 'RADAR-%d.pth' % epoch + '\n')


def val(test_loader, model, epoch, writer):
    """
    validation function
    """
    model.eval()
    tmp_f1 = []
    tmp_mae = []
    with torch.no_grad():


        val_num = test_loader.size
        print('val_num',val_num)
        mybins = np.arange(0, 256)
        PRE = np.zeros((val_num, len(mybins) - 1))
        REC = np.zeros((val_num, len(mybins) - 1))
        F1 = np.zeros((val_num, len(mybins) - 1))
        MAE = np.zeros((val_num))

        for i in range(test_loader.size):
            print('valing',i)
            image, gt, name = test_loader.load_data()
            # print('gt.shape',gt.shape)
            # gt = np.asarray(gt, np.float32)
            # gt /= (gt.max() + 1e-8)
            image = image.cuda()
            pre,sem,edge,_ = model(image)

            #res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            output_size = (gt.shape[1], gt.shape[2])  # Assuming gt is a 4D tensor (batch_size, channels, height, width)
            res = F.interpolate(pre[0], size=output_size, mode='bilinear', align_corners=False).squeeze()
            print('res.shape', res.shape)
            #########
            #res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            gt = np.squeeze(gt)
            gt = gt.cuda()

            if gt.max() == 1:
                gt = gt * 255
            #print('gt.shape', gt.shape)
            pre, rec, f1, mae = f1_mae_torch(res, gt)
           # print('mpre', pre,'rec', rec,'f1', f1,'mae', mae)
        #     mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        # mae = mae_sum / test_loader.size
            PRE[i, :] = pre
            REC[i, :] = rec
            F1[i, :] = f1
            MAE[i] = mae
        PRE_m = np.mean(PRE, 0)
        REC_m = np.mean(REC, 0)
        f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)

        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))
        writer.add_scalar('MAE', tmp_mae, global_step=epoch)
        return tmp_f1,tmp_mae




if __name__ == '__main__':

    writer = SummaryWriter('result' + 'summary')
    if torch.cuda.device_count() >= 2:
        device = torch.device('cuda:1')
    elif torch.cuda.device_count() == 1:
        device = torch.device('cuda:0')
    else:
        device = device = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=704, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./MAS3K_train', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='./MAS3K_test',
                        help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='RADAR')
    opt = parser.parse_args()

    # ---- build models ----
    model = RADAR().cuda()

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/edge_gt/'.format(opt.train_path)

    test_image_root = '{}/Imgs/'.format(opt.test_path)
    test_gt_root = '{}/GT/'.format(opt.test_path)
    #test_edge_root = '{}/edge_gt/'.format(opt.test_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    test_loader = test_dataset(test_image_root, test_gt_root,  testsize=opt.trainsize)

    total_step = len(train_loader)

    print("Start Training:",total_step)

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader,test_loader, model, optimizer, epoch,writer)

    file.close()
