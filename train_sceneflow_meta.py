import datetime
import json
import os
from path import Path
import numpy as np
import torch
import torch.optim
import torch.utils.data
import os
import random
from datasets.sceneflow_sub_folders import SceneflowSubDataset
import torchvision.transforms as transforms
import models
from meta_options import META_options
from meta_algorithm import MetaAlgorithm
from tensorboardX import SummaryWriter
from logger import TermLogger
from utils import save_checkpoint

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


def set_seed(seed = 1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_param(model):
    """
    print number of parameters in the model
    """
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 
                       'encoder.encoder' in n and 'fc' not in n and p.requires_grad)
    print('number of params in encoder:', f'{n_parameters:,}')

    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'decoder.decoder' in n and p.requires_grad)
    print('number of params in decoder:', f'{n_parameters:,}')


def main(args):
    global best_error, n_iter, device

    timestamp = datetime.datetime.now().strftime("%m-%d-%H%M%S")
    save_path = Path(args.name)
    args.save_path = 'checkpoints' / save_path / timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    training_writer = SummaryWriter(args.save_path / 'train')
    valid_writer = SummaryWriter(args.save_path / 'val')

    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        opt_readable = json.dumps(args.__dict__, sort_keys=False, indent=2)
        print("--------------------------------------------------------------------")
        print("TRAINING OPTIONS:")
        print(opt_readable)
        f.write(opt_readable + "\n")
        print("--------------------------------------------------------------------")

    # Data loading code
    debug_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("=> fetching scenes in '{}'".format(args.data))

    debug_set = SceneflowSubDataset(args.data, transform=debug_transform)

    train_len = int(len(debug_set) * 0.8)
    val_len = len(debug_set) - train_len
    train_set, val_set = torch.utils.data.random_split(debug_set, [train_len, val_len])

    print('{} train samples , {} val samples'.format(len(train_set), len(val_set)))

    # 4 * (1 + 1)
    all_batch_size = args.meta_batch_size*(args.q_query + args.k_shot)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=all_batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=all_batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    disp_net= models.SSDNet(args.resnet_layers,  args.with_pretrain, args.max_disp).cuda()

    print('=> setting adam solver') 
    optimizer = torch.optim.Adam(disp_net.parameters(), lr=args.meta_lr,
                                 betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size),
                        valid_size=len(val_loader))   
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss, train_error = train(args, train_loader, disp_net, optimizer, epoch, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}, avg error : {:.3f}'.format(train_loss, train_error))

        training_writer.add_scalar('Epoch train_mean_loss', train_loss, epoch)
        training_writer.add_scalar('Epoch train_mean_err', train_error, epoch)

        # evaluate on validation set
        logger.reset_valid_bar()
        valid_loss, valid_error = valid(args, val_loader, disp_net, optimizer, epoch, logger, valid_writer)
        logger.valid_writer.write(' * Avg Loss : {:.3f}, avg error : {:.3f}'.format(valid_loss, valid_error))

        valid_writer.add_scalar('Epoch valid_mean_loss', valid_loss, epoch)
        valid_writer.add_scalar('Epoch valid_mean_err', valid_error, epoch)

        save_checkpoint(args.save_path, epoch, {'epoch': epoch, 'state_dict': disp_net.state_dict()})

        
def train(args, train_loader, disp_net, optimizer, epoch, logger, train_writer):

    # switch to train mode
    disp_net.train()

    train_meta_loss = []
    train_error = []

    logger.train_bar.update(0)

    intrinsics = []
    intrinsics.append(torch.from_numpy(np.array([[[1050, 0, 479.5],[0, 1050, 269.5],[0, 0, 1]]]).astype(np.float32)).unsqueeze(0).repeat(8, 1, 1, 1))
    intrinsics.append(torch.from_numpy(np.array([[[1050, 0, 479.5],[0, 1050, 269.5],[0, 0, 1]]]).astype(np.float32)).unsqueeze(0).repeat(8, 1, 1, 1))
    pose = torch.from_numpy(np.array([[[ 1.0000,  0.0000,  0.0000, -1],
                                        [ 0.0000,  1.0000,  0.0000,  0.0000],
                                        [ 0.0000,  0.0000,  1.0000,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  1.0000]]]).astype(np.float32)).unsqueeze(0).repeat(8, 1, 1, 1)
    pose_inv = torch.from_numpy(np.array([[[ 1.0000,  0.0000,  0.0000, 1],
                                        [ 0.0000,  1.0000,  0.0000,  0.0000],
                                        [ 0.0000,  0.0000,  1.0000,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  1.0000]]]).astype(np.float32)).unsqueeze(0).repeat(8, 1, 1, 1)

    for i, (tgt_img, ref_img, disp_left, disp_right, mask_L, mask_R) in enumerate(train_loader):

        record_idx = epoch * args.epoch_size + i

        _, c, h, w = tgt_img.shape
        tgt_img = tgt_img.reshape(args.meta_batch_size, -1, c, h, w).to(device)
        ref_img = ref_img.reshape(args.meta_batch_size, -1, c, h, w).to(device)

        _, h, w = disp_left.shape
        disp_left = -disp_left.reshape(args.meta_batch_size, -1, h, w).to(device)
        disp_right = disp_right.reshape(args.meta_batch_size, -1, h, w).to(device)

        # mask_L = mask_L.reshape(args.meta_batch_size, -1, h, w).to(device)
        # mask_R = mask_R.reshape(args.meta_batch_size, -1, h, w).to(device)
        
        left_intrinsics = intrinsics[0].reshape(args.meta_batch_size, -1, 3, 3).to(device)
        right_intrinsics = intrinsics[1].reshape(args.meta_batch_size, -1, 3, 3).to(device)
        pose = pose.reshape(args.meta_batch_size, -1, 4, 4).to(device) 
        pose_inv = pose_inv.reshape(args.meta_batch_size, -1, 4, 4).to(device)

        meta_loss, error = MetaAlgorithm(args, disp_net, optimizer, tgt_img, ref_img, 
                                        disp_left, disp_right, left_intrinsics, right_intrinsics, 
                                        pose, pose_inv, train_writer, record_idx)

        train_writer.add_scalar('train_loss', meta_loss.item(), record_idx)
        train_writer.add_scalar('train_error', error, record_idx)

        train_meta_loss.append(meta_loss.item())
        train_error.append(error)

        logger.train_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Loss {:.3f} Pixel-Error {:.3f}'.format(meta_loss, error))
        if i >= args.epoch_size - 1:
            break

    return np.mean(train_meta_loss), np.mean(train_error)


def valid(args, val_loader, disp_net, optimizer, epoch, logger, valid_writer):

    # switch to train mode
    disp_net.train()

    valid_meta_loss = []
    valid_error = []

    logger.valid_bar.update(0)

    intrinsics = []
    intrinsics.append(torch.from_numpy(np.array([[[1050, 0, 479.5],[0, 1050, 269.5],[0, 0, 1]]]).astype(np.float32)).unsqueeze(0).repeat(8, 1, 1, 1))
    intrinsics.append(torch.from_numpy(np.array([[[1050, 0, 479.5],[0, 1050, 269.5],[0, 0, 1]]]).astype(np.float32)).unsqueeze(0).repeat(8, 1, 1, 1))
    pose = torch.from_numpy(np.array([[[ 1.0000,  0.0000,  0.0000, -1],
                                        [ 0.0000,  1.0000,  0.0000,  0.0000],
                                        [ 0.0000,  0.0000,  1.0000,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  1.0000]]]).astype(np.float32)).unsqueeze(0).repeat(8, 1, 1, 1)
    pose_inv = torch.from_numpy(np.array([[[ 1.0000,  0.0000,  0.0000, 1],
                                        [ 0.0000,  1.0000,  0.0000,  0.0000],
                                        [ 0.0000,  0.0000,  1.0000,  0.0000],
                                        [ 0.0000,  0.0000,  0.0000,  1.0000]]]).astype(np.float32)).unsqueeze(0).repeat(8, 1, 1, 1)

    for i, (tgt_img, ref_img, disp_left, disp_right, mask_L, mask_R) in enumerate(val_loader):
        
        record_idx = epoch * len(val_loader) + i

        _, c, h, w = tgt_img.shape
        tgt_img = tgt_img.reshape(args.meta_batch_size, -1, c, h, w).to(device)
        ref_img = ref_img.reshape(args.meta_batch_size, -1, c, h, w).to(device)

        _, h, w = disp_left.shape
        disp_left = -disp_left.reshape(args.meta_batch_size, -1, h, w).to(device)
        disp_right = disp_right.reshape(args.meta_batch_size, -1, h, w).to(device)

        left_intrinsics = intrinsics[0].reshape(args.meta_batch_size, -1, 3, 3).to(device)
        right_intrinsics = intrinsics[1].reshape(args.meta_batch_size, -1, 3, 3).to(device)
        pose = pose.reshape(args.meta_batch_size, -1, 4, 4).to(device) 
        pose_inv = pose_inv.reshape(args.meta_batch_size, -1, 4, 4).to(device)

        meta_loss, error = MetaAlgorithm(args, disp_net, optimizer, tgt_img, ref_img, 
                                       disp_left, disp_right, left_intrinsics, right_intrinsics, 
                                       pose, pose_inv, valid_writer, record_idx, train=False)

        valid_writer.add_scalar('valid_loss', meta_loss.item(), record_idx)
        valid_writer.add_scalar('valid_error', error, record_idx)

        valid_meta_loss.append(meta_loss.item())
        valid_error.append(error)

        logger.valid_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('Valid: Loss {:.3f} Pixel-Error {:.3f}'.format(meta_loss, error))
        if i >= args.epoch_size - 1:
            break

    return np.mean(valid_meta_loss), np.mean(valid_error)


if __name__ == '__main__':

    args = META_options()

    set_seed(args.seed)

    main(args)
