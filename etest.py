import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models.RADAR import RADAR
from utils.tdataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=1024, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/RADAR/RADAR-99.pth')

for _data_name in ['RADAR']:
    data_path = './MAS3K_test'  #DIS-main/HRUCD4K/test
    save_path = './results/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = RADAR()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+'Edge/', exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        pre,sem, eg, _,  = model(image)
        # print('pre[0].shape',pre[0].shape)
        # print('gt.shape', gt.shape)
        res = F.upsample(pre[0], size=[gt.shape[1],gt.shape[2]], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path+name, (res*255).astype(np.uint8))
        # e = F.upsample(e, size=gt.shape, mode='bilinear', align_corners=True)
        # e = e.data.cpu().numpy().squeeze()
        # e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        # imageio.imwrite(save_path+'edge/'+name, (e*255).astype(np.uint8))