import argparse

from functools import partial
from pathlib import Path


def _type_or_none(arg, typ=str):
    if (arg is None) or (arg.lower() == "none"):
        return None
    return typ(arg)


str_or_none = partial(_type_or_none, typ=str)
int_or_none = partial(_type_or_none, typ=int)
float_or_none = partial(_type_or_none, typ=float)
path_or_none = partial(_type_or_none, typ=Path)


def str_upper(arg):
    if arg is None:
        return None
    return arg.upper()


def str_lower(arg):
    if arg is None:
        return None
    return arg.lower()


def str_to_bool(arg):
    arg = str_lower(arg)
    if arg in {"true", "t", "yes", "y"}:
        return True
    return False


parser = argparse.ArgumentParser()
parser.add_argument('--num-epoch', type=int, default=100,
                        help='number of training epochs')
parser.add_argument('--save-epoch', type=int, default=10,
                        help='frequency of model saving')
parser.add_argument('--train-shot', type=int, default=1,
                        help='number of support examples per training class')
parser.add_argument('--val-shot', type=int, default=1,
                        help='number of support examples per validation class')
parser.add_argument('--train-query', type=int, default=15,
                        help='number of query examples per training class')
parser.add_argument('--val-episode', type=int, default=600,
                        help='number of episodes per validation')
parser.add_argument('--val-query', type=int, default=15,
                        help='number of query examples per validation class')
parser.add_argument('--train-way', type=int, default=5,
                        help='number of classes in one training episode')
parser.add_argument('--test-way', type=int, default=5,
                        help='number of classes in one test (or validation) episode')
parser.add_argument('--save-path', default='./experiments/meta_part_resnet12_mini_counterfactul')
parser.add_argument('--gpu', default='0')
parser.add_argument('--network', type=str, default='ResNet12',
                        help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
parser.add_argument('--head', type=str, default='FuseCosNet',
                        help='choose which classification head to use. FuseCosNet, classification(cls)head')
parser.add_argument('--pre_head', type=str, default='LinearNet',
                        help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
parser.add_argument('--dataset', type=str, default='miniImageNet',
                        help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
parser.add_argument('--episodes-per-batch', type=int, default=8,
                        help='number of episodes per batch')
parser.add_argument('--eps', type=float, default=0.0,
                        help='epsilon of label smoothing')
parser.add_argument('--phase', type=str, default='metatest',
                    help='')
parser.add_argument('--seed', type=int, default=45,help='number of episodes per batch')
parser.add_argument('--attSize', type=int, default=171,help='number of attribute feature size')
parser.add_argument('--latent_size', type=int, default=256,help='number of netE output size')
parser.add_argument('--feature_size', type=int, default=256,help='number of embedding_net output size')
parser.add_argument('--ndh', type=int, default=4096,help='size of the hidden units in discriminator')
parser.add_argument('--ngh', type=int, default=4096,help='size of the hidden units in discriminator')
parser.add_argument('--feedback_loop', type=int, default=2,help='feedback loop iterations')
parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
parser.add_argument('--gammaD', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument('--gammaG', type=int, default=10, help='weight on the W-GAN loss')
parser.add_argument("--encoder_use_y", action="store_true", default=False, help="Encoder use y as input")
parser.add_argument("--general", action="store_true", default=False, help="use prior Y or not")
parser.add_argument("--distill", action="store_true", default=False, help="use distill version of model")
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--recon", type=str, default="l2", help="VAE reconstruction loss: bce or l2 or l1")
parser.add_argument("--generative_model", type=str, default="vae", help="VAE,vaegan")
parser.add_argument("--scaler_filename", type=str, default="trainval_MinMax.save", help="path to feature scaler")
parser.add_argument("--z_disentangle", action="store_true", default=False, help="Use z disentangle loss")
parser.add_argument("--zd_tcvae", action="store_true", default=False, help="Use TCVAE")
parser.add_argument("--baseline", action="store_true", default=False, help="run baseline results(class attribute is set to 0)")
parser.add_argument("--zd_beta_annealing", action="store_true", default=False, help="Slowly increase beta")
parser.add_argument("--zd_beta", type=float, default=1.0, help="beta for scaling KL loss")
parser.add_argument("--add_noise", type=float, default=0.0, help="Add noise to reconstruction while training")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for contrastive loss")
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--recons_weight', type=float, default=1e-2, help='recons_weight for decoder')
parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')
parser.add_argument("--contra_lambda", type=float, default=1.0, help="Scaling factor of contrastive loss")
parser.add_argument('--a1', type=float, default=1.0)
parser.add_argument('--a2', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate to train GANs ')
parser.add_argument('--feed_lr', type=float, default=1e-5, help='learning rate to train GANs ')
parser.add_argument('--dec_lr', type=float, default=1e-4, help='learning rate to train GANs ')
parser.add_argument('--gem', action='store_true', help='using generalizedMeanPooling')
parser.add_argument('--norm', action='store_true', help='using feature l2 norm')
parser.add_argument('--use_feature', action='store_true', help='using direct features')

opt = parser.parse_args()

opt.encoder_layer_sizes[0] = opt.feature_size
opt.decoder_layer_sizes[-1] = opt.feature_size