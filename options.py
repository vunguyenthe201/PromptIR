import argparse
import logging
import os
import os.path as osp
import sys
import math
import yaml
from collections import OrderedDict
try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

def OrderedYaml():
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

Loader, Dumper = OrderedYaml()

def parse(opt_path, is_train=True):
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    opt["is_train"] = is_train

    scale = 1
    if opt['distortion'] == 'sr':
        scale = opt['degradation']['scale']

        ##### sr network ####
        opt["network_G"]["setting"]["upscale"] = scale
        # opt["network_G"]["setting"]["in_nc"] *= scale**2

    # datasets
    for phase, dataset in opt["datasets"].items():
        phase = phase.split("_")[0]
        print(dataset)
        dataset["phase"] = phase
        dataset["scale"] = scale
        dataset["distortion"] = opt["distortion"]
        
        is_lmdb = False
        if dataset.get("dataroot_GT", None) is not None:
            dataset["dataroot_GT"] = osp.expanduser(dataset["dataroot_GT"])
            if dataset["dataroot_GT"].endswith("lmdb"):
                is_lmdb = True
        # if dataset.get('dataroot_GT_bg', None) is not None:
        #     dataset['dataroot_GT_bg'] = osp.expanduser(dataset['dataroot_GT_bg'])
        if dataset.get("dataroot_LQ", None) is not None:
            dataset["dataroot_LQ"] = osp.expanduser(dataset["dataroot_LQ"])
            if dataset["dataroot_LQ"].endswith("lmdb"):
                is_lmdb = True
        dataset["data_type"] = "lmdb" if is_lmdb else "img"
        if dataset["mode"].endswith("mc"):  # for memcached
            dataset["data_type"] = "mc"
            dataset["mode"] = dataset["mode"].replace("_mc", "")

    # path
    for key, path in opt["path"].items():
        if path and key in opt["path"] and key != "strict_load":
            opt["path"][key] = osp.expanduser(path)
    opt["path"]["root"] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir)
    )
    path = osp.abspath(__file__)
    config_dir = path.split("/")[-2]
    if is_train:
        experiments_root = osp.join(
            opt["path"]["root"], "experiments", config_dir, opt["name"]
        )
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = osp.join(experiments_root, "models")
        opt["path"]["training_state"] = osp.join(experiments_root, "training_state")
        opt["path"]["log"] = experiments_root
        opt["path"]["val_images"] = osp.join(experiments_root, "val_images")

        # change some options for debug mode
        if "debug" in opt["name"]:
            opt["train"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        results_root = osp.join(opt["path"]["root"], "results", config_dir)
        opt["path"]["results_root"] = osp.join(results_root, opt["name"])
        opt["path"]["log"] = osp.join(results_root, opt["name"])

    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
    
    
parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument("--opt", type=str, help="Path to option YMAL file.")


parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=1,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default=4,help = "Number of GPUs to use for training")

options = parser.parse_args()
options.batch_size = len(options.de_type)

opt_dict = parse(options.opt, is_train=True)
opt_dict = dict_to_nonedict(opt_dict)