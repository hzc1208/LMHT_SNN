from utils import *
import argparse
from ann_models.ResNet import *
from ann_models.VGG import *
import torch
import random
import os
import numpy as np
import logging
from dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, PreProcess_TinyImageNet
from torch.cuda import amp
from timm.data import Mixup


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def init_distributed(distributed_init_mode):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    print('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode,
                                         world_size=world_size, rank=rank)
    return True, rank, world_size, local_rank
    
    
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt
    

def train_one_epoch(model, loss_fn, optimizer, train_dataloader, local_rank, scaler=None, mixup=None, distributed=False):
    epoch_loss, lenth = 0, 0
    model.train()
    for img, label in train_dataloader:
        img = img.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)
        lenth += len(img)
        optimizer.zero_grad()

        if mixup:
            img, label = mixup(img, label)
        
        if scaler is not None:
            with amp.autocast():
                spikes = model(img).mean(dim=0)
                loss = loss_fn(spikes, label)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:     
            spikes = model(img).mean(dim=0)
            loss = loss_fn(spikes, label)
            loss.backward()
            optimizer.step()
            
        if distributed:
            vis_loss = reduce_mean(loss, torch.distributed.get_world_size())
            epoch_loss += vis_loss.item()
        else:
            epoch_loss += loss.item()
    
    return epoch_loss/lenth


def eval_one_epoch(model, test_dataloader, sim_len):
    tot = torch.zeros(sim_len).cuda()
    model.eval()
    lenth = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            spikes = 0
            img = img.to(torch.device('cuda'), non_blocking=True)
            label = label.to(torch.device('cuda'), non_blocking=True)
            lenth += len(img)
            out = model(img)
            for t in range(sim_len):
                spikes += out[t]
                tot[t] += (label==spikes.max(1)[1]).sum().item()
      
    return tot/lenth


def train_ann_one_epoch(model, loss_fn, optimizer, train_dataloader, local_rank, scaler=None, mixup=None, distributed=False):
    epoch_loss, lenth = 0, 0
    model.train()
    for img, label in train_dataloader:
        img = img.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)
        lenth += len(img)
        optimizer.zero_grad()

        if mixup:
            img, label = mixup(img, label)
        
        if scaler is not None:
            with amp.autocast():
                spikes = model(img)
                loss = loss_fn(spikes, label)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:     
            spikes = model(img)
            loss = loss_fn(spikes, label)
            loss.backward()
            optimizer.step()
            
        if distributed:
            vis_loss = reduce_mean(loss, torch.distributed.get_world_size())
            epoch_loss += vis_loss.item()
        else:
            epoch_loss += loss.item()
    
    return epoch_loss/lenth


def eval_ann_one_epoch(model, test_dataloader):
    model.eval()
    acc, lenth = 0, 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.to(torch.device('cuda'), non_blocking=True)
            label = label.to(torch.device('cuda'), non_blocking=True)
            lenth += len(img)
            out = model(img)
            acc += (label==out.max(1)[1]).sum().item()
      
    return acc/lenth


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='/home/cifar100/', help='Directory where the dataset is saved')
    parser.add_argument('--savedir', type=str, default='/home/', help='Directory where the model is saved')
    parser.add_argument('--trainann_epochs', type=int, default=300, help='Training Epochs of ANNs')
    parser.add_argument('--trainsnn_epochs', type=int, default=30, help='Training Epochs of SNNs')
    parser.add_argument('--net_arch', type=str, default='resnet34', help='Network Architecture')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size')
    parser.add_argument('--time_step', type=int, default=2, help='Training Time-steps for SNNs')
    parser.add_argument('--thre_level', type=int, default=2, help='Threshold Level')
    parser.add_argument('--qcfs_level', type=int, default=8, help='QCFS ANN Level')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--lr2', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--direct_inference', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dev', type=str, default='0')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--distributed_init_mode', type=str, default='env://')
    parser.add_argument("--sync_bn", action="store_true", help="Use sync batch norm")
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--pretrained_ann_path', type=str, default='')
    parser.add_argument('--mixup', action='store_true', help='Mixup')
    parser.add_argument('--amp', action='store_true', help='Use AMP training')
    parser.add_argument('--warm-up', type=str, nargs='+', default=[], help='--warm-up <epochs> <start-factor>')
    
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
    
    torch.backends.cudnn.benchmark = True
    _seed_ = args.seed
    random.seed(_seed_)
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)
    
    log_dir = args.savedir + args.dataset + '-hybrid-' + args.net_arch + '-Q' + str(args.qcfs_level) + '-L' + str(args.thre_level) + '-T' + str(args.time_step)
    identifier = 'lr' + str(args.lr2) + '_wd' + str(args.wd) + '_epoch' + str(args.trainsnn_epochs) + '_amp' + str(args.amp)
    save_name_suffix = log_dir + '/' + identifier
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    
    distributed, rank, world_size, local_rank = init_distributed(args.distributed_init_mode)

    if args.dataset == 'CIFAR10':
        train_dataloader, test_dataloader, train_sampler, test_sampler = PreProcess_Cifar10(args.datadir, args.batchsize, distributed)
        train_ann_dataloader, test_ann_dataloader, _, _ = PreProcess_Cifar10(args.datadir, args.batchsize, False)
        cls = 10
    elif args.dataset == 'CIFAR100':
        train_dataloader, test_dataloader, train_sampler, test_sampler = PreProcess_Cifar100(args.datadir, args.batchsize, distributed)
        train_ann_dataloader, test_ann_dataloader, _, _ = PreProcess_Cifar100(args.datadir, args.batchsize, False)
        cls = 100
    elif args.dataset == 'TinyImageNet':
        train_dataloader, test_dataloader, train_sampler, test_sampler = PreProcess_TinyImageNet(args.datadir, args.batchsize, distributed)
        train_ann_dataloader, test_ann_dataloader, _, _ = PreProcess_TinyImageNet(args.datadir, args.batchsize, False)
        cls = 200
    else:
        error('unable to find dataset ' + args.dataset)

        
    if args.net_arch == 'resnet20':
        model = resnet20(num_classes=cls)
    elif args.net_arch == 'vgg13':
        model = vgg13(num_classes=cls)
    elif args.net_arch == 'vgg16':
        model = vgg16(num_classes=cls) 
    else:
        error('unable to find model ' + args.net_arch)
        
    model = replace_activation_by_QCFS(model, args.qcfs_level, 8.)
    
    if local_rank == 0:
        print(model)
    
    model.cuda()
        
    mixup = None
    if args.mixup:
        mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=1.0,
                      switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=cls)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None  
        
    ann_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    ann_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ann_optimizer, T_max=args.trainann_epochs)
    ann_loss_fn = nn.CrossEntropyLoss()
    
    if len(args.pretrained_ann_path) > 0 and local_rank == 0:
        model.load_state_dict(torch.load(args.pretrained_ann_path, map_location='cpu')) #['model']
        acc = eval_ann_one_epoch(model, test_ann_dataloader)
        logger.info(f"Pretrained ANNs Test Acc: {acc}")
    elif local_rank == 0:
        best_acc = 0.
        for epoch in range(0, args.trainann_epochs):
            epoch_loss = train_ann_one_epoch(model, ann_loss_fn, ann_optimizer, train_ann_dataloader, local_rank, scaler, mixup, False)
            ann_scheduler.step()

            acc = eval_ann_one_epoch(model, test_ann_dataloader)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': ann_optimizer.state_dict(),
                'scheduler': ann_scheduler.state_dict(),
                'epoch': epoch,
                'max_acc1': acc
                }
            if best_acc < acc:
                best_acc = acc
                torch.save(checkpoint, save_name_suffix + '_best_ann_checkpoint.pth')
            torch.save(checkpoint, save_name_suffix + '_current_ann_checkpoint.pth')

            logger.info(f"ANNs training Epoch {epoch}: Val_loss: {epoch_loss}")
            logger.info(f"ANNs training Epoch {epoch}: Test Acc: {acc} Best Acc: {best_acc}")
            
        model.load_state_dict(torch.load(save_name_suffix + '_best_ann_checkpoint.pth', map_location='cpu')['model'])
            
    if distributed:
        torch.distributed.barrier()     
    if local_rank == 0:
        logger.info(f"\n=== SNNs Training Begin ===\n")
        
    del ann_optimizer, ann_scheduler, ann_loss_fn, train_ann_dataloader, test_ann_dataloader
    
    model = replace_layer_by_snn_layer(model)
    model = replace_QCFS_by_IFNode(model, 16)
    model.ann_mode = False
    model.T = 16
    model.cuda()
    if local_rank == 0:
        print(model)
        acc = eval_one_epoch(model, test_dataloader, 16)
        logger.info(f"ANN2SNN Inference: Test Acc: {acc}")
    
    model = replace_IFNode_by_LMHT(model, args.thre_level, args.time_step)
    model.T = args.time_step
    model.cuda()
    
    if args.net_arch == 'resnet20':
        model.conv1[2].scale = args.thre_level
    elif args.net_arch == 'vgg16':
        model.layer1[2].scale = args.thre_level
    
    if local_rank == 0:
        print(model)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr2, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.trainsnn_epochs)
    loss_fn = nn.CrossEntropyLoss()
    
    if len(args.warm_up) != 0:
        assert len(args.warm_up) == 2
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                              start_factor=float(args.warm_up[1]),
                                              total_iters=int(args.warm_up[0])),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.trainsnn_epochs-int(args.warm_up[0])), ])

    model_without_ddp = model
    
    if distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_acc1 = checkpoint['max_acc1']
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(max_acc1, start_epoch)
    else:
        start_epoch = 0
        max_acc1 = 0


    if args.direct_inference is not True:
        best_acc = max_acc1
        
        for epoch in range(start_epoch, args.trainsnn_epochs):
            if distributed:
                train_sampler.set_epoch(epoch)
            epoch_loss = train_one_epoch(model, loss_fn, optimizer, train_dataloader, local_rank, scaler, mixup, distributed)
            scheduler.step()

            if local_rank == 0:
                acc = eval_one_epoch(model, test_dataloader, args.time_step)
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'max_acc1': acc[-1].item()
                    }
                if best_acc < acc[-1].item():
                    best_acc = acc[-1].item()
                    torch.save(checkpoint, save_name_suffix + '_best_checkpoint.pth')
                torch.save(checkpoint, save_name_suffix + '_current_checkpoint.pth')

                logger.info(f"SNNs training Epoch {epoch}: Val_loss: {epoch_loss}")
                logger.info(f"SNNs training Epoch {epoch}: Test Acc: {acc} Best Acc: {best_acc}")
            
            if distributed:
                torch.distributed.barrier()
                
    else:
        if local_rank == 0:
            print(f'=== Load Pretrained SNNs ===')
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model']) 
            new_acc = eval_one_epoch(model, test_dataloader, args.time_step)
            print(new_acc)   