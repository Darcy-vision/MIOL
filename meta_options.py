import argparse

def META_options():
    parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data', required=True, default='path_to_SceneFlowSubData',
                        metavar='DIR', help='path to Sceneflow dataset')
    parser.add_argument('--name', required=True, dest='name', type=str, default='meta_train',
                    help='name of the experiment, checkpoints are stored in checpoints/name')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers')
    parser.add_argument('--epochs', default=10, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                        help='manual epoch size (will match dataset size if not set)')
    parser.add_argument('--meta-batch-size', default=4,
                        type=int, metavar='N', help='meta-batch size')
    parser.add_argument('-k', '--k-shot', default=1,
                        type=int, metavar='N', help='train-data size')
    parser.add_argument('-q', '--q-query', default=1,
                        type=int, metavar='N', help='test-data size')
    parser.add_argument('--inner-lr', default=1e-4, type=float, metavar='LR', help='inner loop learning rate')
    parser.add_argument('--meta-lr', default=1e-4, type=float, metavar='LR', help='outer loop learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        metavar='M', help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float,
                        metavar='M', help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0,
                        type=float, metavar='W', help='weight decay')
    parser.add_argument('--print-freq', default=1, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for random functions, and network initialization')
    parser.add_argument('--resnet-layers', type=int, default=18,
                        choices=[18, 50], help='number of ResNet layers for depth estimation.')
    parser.add_argument('--max-disp', type=int,
                        default=192, help='max disparity of network')
    parser.add_argument('--num-scales', '--number-of-scales',
                        type=int, help='the number of scales', metavar='W', default=1)
    parser.add_argument('-p', '--photo-loss-weight', type=float,
                        help='weight for photometric loss', metavar='W', default=1)
    parser.add_argument('-s', '--smooth-loss-weight', type=float,
                        help='weight for disparity smoothness loss', metavar='W', default=0.1)
    parser.add_argument('-c', '--geometry-consistency-weight', type=float,
                        help='weight for depth consistency loss', metavar='W', default=0.5)
    parser.add_argument('--with-ssim', type=int,
                        default=1, help='with ssim or not')
    parser.add_argument('--with-mask', type=int, default=1,
                        help='with the the mask for moving objects and occlusions or not')
    parser.add_argument('--with-auto-mask', type=int, default=0,
                        help='with the the mask for stationary points')
    parser.add_argument('--with-pretrain', type=int, default=1,
                        help='with or without imagenet pretrain for resnet')
    parser.add_argument('--pretrained-disp', dest='pretrained_disp',
                        default=None, metavar='PATH', help='path to pre-trained dispnet model')
    parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                        help='padding mode for image warping : this is important for photometric differenciation when going outside target image. zeros will null gradients outside target image.'
                            ' border will only null gradients of the coordinate outside (x or y)')

    args = parser.parse_args()

    return args