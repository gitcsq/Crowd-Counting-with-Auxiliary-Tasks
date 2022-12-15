import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from datasets.crowd_count import Crowd_qnrf, Crowd_nwpu, Crowd_sh
from Networks.VGG import VGG16
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
import tqdm

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    gt_counts = torch.stack(transposed_batch[1], 0)
    return images, gt_counts


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = 'VGG16/STA-1-11-input-{}'.format(args.crop_size)

        self.save_dir = os.path.join('ckpts', sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        downsample_ratio = 8
        if args.dataset.lower() == 'qnrf':
            self.datasets = {x: Crowd_qnrf(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'nwpu':
            self.datasets = {x: Crowd_nwpu(os.path.join(args.data_dir, x),
                                           args.crop_size, downsample_ratio, x) for x in ['train', 'val']}
        elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
            self.datasets = {'train': Crowd_sh(os.path.join(args.data_dir, 'train_data'),
                                               args.crop_size, downsample_ratio, 'train'),
                             'val': Crowd_sh(os.path.join(args.data_dir, 'test_data'),
                                             args.crop_size, downsample_ratio, 'val'),
                             }
        else:
            raise NotImplementedError

        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers * self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model = VGG16.VGG()
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.start_epoch = 0
        if args.resume:
            self.logger.info('loading pretrained model from ' + args.resume)
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))
        else:
            self.logger.info('random initialization')

        #self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.SmoothL1Loss(reduction='none').to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = 80
        self.best_mse = np.inf
        #self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        #epoch_mse_loss = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        for step, (inputs, gt_counts) in enumerate(tqdm.tqdm(self.dataloaders['train'])):
            inputs = inputs.to(self.device)
            gt_counts = gt_counts.unsqueeze(1).to(self.device)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                outputs = outputs.sum(2).sum(2)

                count_loss = self.mae(outputs, gt_counts)
                # print('********', outputs.transpose(0,1).detach().cpu().numpy(), gt_counts.transpose(0,1).detach().cpu().numpy())
                # print('********', count_loss.transpose(0,1).detach().cpu().numpy())
                weight = torch.abs(outputs-gt_counts) / (gt_counts+1e-5)
                t_one = torch.ones_like(weight, dtype=torch.float)
                weight = torch.where(weight > 1.0, t_one, weight)
                weight = weight / weight.sum()
                # print('--------', weight.transpose(0,1).detach().cpu().numpy())
                count_loss = (count_loss * weight).sum()
                # print('********', count_loss.transpose(0,1).detach().cpu().numpy(), weight.transpose(0,1).detach().cpu().numpy())

                epoch_count_loss.update(count_loss.item(), N)

                loss = count_loss
                epoch_loss.update(loss.item(), N)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                gt_counts = gt_counts.detach().cpu().numpy()
                pred_err = pred_count - gt_counts

                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

        self.logger.info(
            'Epoch {} Train, Loss: {:.2f}, Count Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                .format(self.epoch, epoch_loss.get_avg(), epoch_count_loss.get_avg(), np.sqrt(epoch_mse.get_avg()),
                        epoch_mae.get_avg(), time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)

    def val_epoch(self):
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        for inputs, count, name in tqdm.tqdm(self.dataloaders['val']):
            with torch.no_grad():
                #nputs = cal_new_tensor(inputs, min_size=args.crop_size)
                inputs = inputs.to(self.device)
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()
                rh, rw = args.crop_size, args.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

                crop_preds = []
                nz, bz = crop_imgs.size(0), args.batch_size
                for i in range(0, nz, bz):
                    gs, gt = i, min(nz, i + bz)
                    crop_pred = self.model(crop_imgs[gs:gt])

                    _, _, h1, w1 = crop_pred.size()
                    crop_pred = F.interpolate(crop_pred, size=(h1 * 8, w1 * 8), mode='bilinear',
                                              align_corners=True) / 64

                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)

                # splice them to the original size
                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1
                # for the overlapping area, compute average value
                mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.logger.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                         .format(self.epoch, mse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        # if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
        if mae < self.best_mae:
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                     self.best_mae,
                                                                                     self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_mae {:.2f}.pth'.format(self.best_mae)))
            # torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            #self.best_count += 1


if __name__ == '__main__':
    import torch
    print(torch.__file__)
    x = torch.ones(1, 3, 768, 1152)