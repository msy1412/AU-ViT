from __future__ import print_function

import os
import sys
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils import data
from data import RAFAU,RAFAU_100
from data import FERPlus
from utils import AverageMeter, accuracy, AU_metric, EXPR_metric
from models import IR_RVT_AU_plus_patch
from utils import parameter
from torch.optim.lr_scheduler import _LRScheduler

FER_RATIO=1
torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)

def parse_arguments():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--CUDA', type=str, default="0,1")
    parser.add_argument('--AU_patch', type=int, default=2)
    parser.add_argument('--print_freq', type=int, default=15)
    parser.add_argument('--AU_rate', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate') 
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--scheduler', type=str ,default="cosin",
                        help='scheduler: cosin or step or none')


    # model dataset
    parser.add_argument('--model', type=str, default='IR50RVT_AU_Patch')
    parser.add_argument('--dataset', type=str, default='FERPlus')
    # other setting

    parser.add_argument('--save_model', type=bool,default=True,
                        help='save model')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.CUDA
    args.model_path = './save/{}_models'.format(args.dataset)
    args.model_name = '{}_{}_{}_{}_{}'.\
        format(args.dataset, args.model, args.learning_rate, args.batch_size,args.scheduler)
    print(f'model name: {args.model_name}')

    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    args.n_cls = [8, 21]
    return args


def set_loader(args):

    # construct data loader

    mean = (0.5,0.5,0.5)
    std = (0.5, 0.5,0.5)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize(size=(112, 112)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(0.5),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(112, 112)),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == 'FERPlus':
        AU_img_folder_path=parameter.AU_img_folder_path
        AU_train_list_file = parameter.AU_train_list_file
        AU_test_list_file = parameter.AU_test_list_file
        
        AU_train_dataset =  RAFAU_100.load_RAFAU(AU_train_list_file,AU_img_folder_path,transform=train_transform,phase="train")
        AU_train_loader = data.DataLoader(AU_train_dataset,batch_size=64, shuffle=True,num_workers=args.num_workers, pin_memory=True)
        AU_test_dataset =  RAFAU_100.load_RAFAU(AU_test_list_file,AU_img_folder_path,transform=val_transform,phase="train")
        AU_test_loader = data.DataLoader(AU_test_dataset,batch_size=42, shuffle=True,num_workers=args.num_workers, pin_memory=True)

        FERPlus_img_folder_path =parameter.FERPlus__img_folder_path

        FERPlus_Traindataset = FERPlus.FERPlus(FERPlus_img_folder_path, 'train',transform=train_transform)
        TrainLoader=data.DataLoader(FERPlus_Traindataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
        
        FERPlus_TestDataset = FERPlus.FERPlus(FERPlus_img_folder_path, 'test',transform=val_transform)#, mode='probability'
        TestLoader=data.DataLoader(FERPlus_TestDataset,batch_size=args.batch_size*3,num_workers=args.num_workers,shuffle=True)
        
        print('AU_Train set size:', AU_train_loader.__len__())
        print('AU_Validation set size:', AU_test_loader.__len__())
        print('FER_Train set size:', TrainLoader.__len__())
        print('FER_Validation set size:', TestLoader.__len__())
        # train_sampler = weighted_sampler_generator(data_txt_dir, args.dataset)
        # train_sampler = None
    else:
        raise ValueError(args.dataset)

    return AU_train_loader,AU_test_loader, TrainLoader,TestLoader

def set_model(args):
    pthpath="/home/sztu/msy_Project/save_model/backbone_ir50_ms1m_epoch63.pth"
    model =IR_RVT_AU_plus_patch.IR50_ViT(num_classes=8,ir_50_pth=pthpath,num_AU_patch=args.AU_patch)


    criterion_EXPR = torch.nn.CrossEntropyLoss()
    pos_weight =np.array([ 3.04216867,  4.40402685,  1.42384106,  3.5033557 ,  8.63157895,
       11.74050633,  4.62290503,  2.0407855 ,  2.38603869, 40.9375    ,
       14.66536965,  4.98216939,  7.37006237, 39.66666667, 19.02985075,
       11.94533762, 38.86138614, 20.18947368,  0.52442257,  2.96259843,
        4.27653997])
    pos_weight=torch.from_numpy(pos_weight)
    criterion_AU = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion_EXPR = criterion_EXPR.cuda()
        criterion_AU = criterion_AU.cuda()
        cudnn.benchmark = True

    return model, [criterion_EXPR, criterion_AU]


def set_optimizer(opt, model):

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                               lr=opt.learning_rate,
                               weight_decay=opt.weight_decay)
    else:
        raise ValueError('optimizer not supported: {}'.format(opt.optimizer))

    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def train(AU_train_loader,FER_train_loader, model, criterion, optimizer,warmup_scheduler, epoch, warmup_epoch, args):
    """one epoch training"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}
    iter_AU_loader=iter(AU_train_loader)
    end = time.time()

    for idx, (images, FER_targets) in enumerate(FER_train_loader):
        data_time.update(time.time() - end)

        # load AU data
        try:
          AU_img,AU_targets=next(iter_AU_loader)
        except:
          iter_AU_loader=iter(AU_train_loader)
          AU_img,AU_targets=next(iter_AU_loader)
        # concat FER and AU imgs
        total_img=torch.cat((images,AU_img),dim=0)
        # process FER labels
        FER_targets = FER_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        FER_targets = torch.as_tensor(FER_targets, dtype=torch.int64)
        FER_targets= FER_targets.cuda()
        # process AU labels
        AU_target_arr = np.array(AU_targets,dtype='int32').T
        AU_target_tensor = torch.tensor(AU_target_arr)
        AU_targets = AU_target_tensor.cuda()
        #batch-size
        FER_bsz = len(FER_targets)
        AU_bsz = len(AU_targets)
        whole_bsz=FER_bsz+AU_bsz

        # imgs to cuda
        total_img = total_img.cuda()
        # model
        output = model(total_img)

        FER_output = output[0][0:FER_bsz]
        AU_output = output[1][FER_bsz:]
        #AU_output = nn.Parameter(torch.ones(AU_bsz, 21)).cuda()
        loss_EXPR = criterion[0](FER_output, FER_targets)
        loss_AU = criterion[1](AU_output, AU_targets.float())
        loss = FER_RATIO * loss_EXPR + args.AU_rate * loss_AU


        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric
        losses.update(loss, whole_bsz)
        acc_EXPR_batch = accuracy(FER_output, FER_targets)
        acc_EXPR.update(acc_EXPR_batch[0], FER_bsz)
        label_EXPR['gt'].append(FER_targets.cpu().detach().numpy())
        label_EXPR['pred'].append(FER_output.cpu().detach().numpy())
        predict_AU = torch.sigmoid(AU_output)
        predict_AU = torch.round(predict_AU)
        correct_sum = sum(predict_AU == AU_targets).sum()
        acc_AU_batch = correct_sum.float()/(AU_bsz*args.n_cls[1])
        acc_AU.update(acc_AU_batch, AU_bsz)
        label_AU['gt'].append(AU_targets.cpu().detach().numpy())
        label_AU['pred'].append(predict_AU.cpu().detach().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch <= warmup_epoch and args.warmup and (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                  'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'.format(
                epoch, idx + 1, len(FER_train_loader),
                # batch_time=batch_time,data_time=data_time, 
                loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU))
            sys.stdout.flush()
        elif (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                  'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'
                  'lr {lr:.8f}\t'.format(
                epoch, idx + 1, len(FER_train_loader),
                # batch_time=batch_time,data_time=data_time, 
                loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR, acc_AU=acc_AU))
            sys.stdout.flush()

    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]
    label_gt = np.concatenate(label_AU['gt'], axis=0)
    label_pred = np.concatenate(label_AU['pred'], axis=0)
    f1, acc, total_acc = AU_metric(label_pred, label_gt)
    AU_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs, AU_accs


def validate(AU_val_loader,FER_val_loader, model, criterion, args):
    """validation"""
    model.eval()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}
    iter_AU_loader=iter(AU_val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, (images, FER_targets) in enumerate(FER_val_loader):
            data_time.update(time.time() - end)
            # load AU data
            try:
                AU_img,AU_targets=next(iter_AU_loader)
            except:
                iter_AU_loader=iter(AU_val_loader)
                AU_img,AU_targets=next(iter_AU_loader)
            # concat FER and AU imgs
            total_img=torch.cat((images,AU_img),dim=0)
            # process FER labels
            FER_targets = FER_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
            FER_targets = torch.as_tensor(FER_targets, dtype=torch.int64)
            FER_targets= FER_targets.cuda()
            # process AU labels
            AU_target_arr = np.array(AU_targets,dtype='int32').T
            AU_target_tensor = torch.tensor(AU_target_arr)
            AU_targets = AU_target_tensor.cuda()
            #batch-size
            FER_bsz = len(FER_targets)
            AU_bsz = len(AU_targets)
            whole_bsz=FER_bsz+AU_bsz
    
            # imgs to cuda
            total_img = total_img.cuda()
            # model
            output = model(total_img)
            FER_output = output[0][0:FER_bsz]
            AU_output = output[1][FER_bsz:]
            #AU_output = nn.Parameter(torch.ones(AU_bsz, 21)).cuda()
            loss_EXPR = criterion[0](FER_output, FER_targets)
            loss_AU = criterion[1](AU_output, AU_targets.float())
            loss = FER_RATIO * loss_EXPR + args.AU_rate * loss_AU
    
    
            # update metric
            losses.update(loss, whole_bsz)
            acc_EXPR_batch = accuracy(FER_output, FER_targets)
            acc_EXPR.update(acc_EXPR_batch[0], FER_bsz)
            label_EXPR['gt'].append(FER_targets.cpu().detach().numpy())
            label_EXPR['pred'].append(FER_output.cpu().detach().numpy())
            predict_AU = torch.sigmoid(AU_output)
            predict_AU = torch.round(predict_AU)
            correct_sum = sum(predict_AU == AU_targets).sum()
            acc_AU_batch = correct_sum.float()/(AU_bsz*args.n_cls[1])
            acc_AU.update(acc_AU_batch, AU_bsz)
            label_AU['gt'].append(AU_targets.cpu().detach().numpy())
            label_AU['pred'].append(predict_AU.cpu().detach().numpy())
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if (idx + 1) % args.print_freq == 0:
                print('test: [{0}/{1}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                      'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'.format(
                    idx + 1, len(FER_val_loader), batch_time=batch_time,
                    loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU))
                sys.stdout.flush()

    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]
    label_gt = np.concatenate(label_AU['gt'], axis=0)
    label_pred = np.concatenate(label_AU['pred'], axis=0)
    f1, acc, total_acc = AU_metric(label_pred, label_gt)
    AU_accs = [f1, acc, total_acc]


    return losses.avg, EXPR_accs, AU_accs

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
       
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class Logger(object):
    def __init__(self, filename="log.txt"):
        import time
        localtime = time.localtime(time.time())
        f = ""
        for i in range(0,5):
            f += str(localtime[i]) + "_"
        filename="log/"+f+filename
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

def main():
    args = parse_arguments()
    sys.stdout = Logger()
    print(args)
    # build data loader
    AU_train_loader,AU_test_loader,FER_train_loader, FER_val_loader = set_loader(args)

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(args, model)
    warmup_scheduler=None
    warmup_epoch=0
    if args.warmup!=0:
        warmup_epoch = args.warmup
        warmup_scheduler = WarmUpLR(optimizer, warmup_epoch)

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
        print("scheduler : multistep")
    elif args.scheduler == 'cosin':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epoch-2,eta_min=5e-7)
        print("scheduler : cosin")

    # training routine
    for epoch in range(1, args.epochs + 1):
        #warm up
        if args.warmup and epoch <= warmup_epoch :
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()
            print("warm_lr:%s" % round(warm_lr[0],5))
        # train for one epoch
        time1 = time.time()
        loss, EXPR_accs, AU_accs = train(AU_train_loader,FER_train_loader, model, criterion, optimizer,warmup_scheduler, epoch,warmup_epoch, args)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}, AU F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
            epoch, time2 - time1, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2], AU_accs[0], AU_accs[1], AU_accs[2]))

        # eval for one epoch
        time1 = time.time()
        loss, EXPR_accs, AU_accs= validate(AU_test_loader,FER_val_loader, model, criterion, args)
        time2 = time.time()
        print('Validation epoch {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}, AU F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
            epoch, time2 - time1, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2], AU_accs[0], AU_accs[1], AU_accs[2]))
        
        if args.scheduler:
            if args.warmup and epoch < warmup_epoch:
                continue
            scheduler.step()  # update lr
            
        if args.save_model:
            if EXPR_accs[1]>= 0.90:
                print("save weights")
                save_file = os.path.join(
                    args.save_folder, '{patch}_epoch_{epoch}_{acc}.pth'.format(patch=args.AU_patch,epoch=epoch,acc=round(EXPR_accs[1],5)))
                save_model(model, optimizer, args, epoch, save_file)


if __name__ == '__main__':
    main()
