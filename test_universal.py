import argparse
import logging
import os
import os.path
import random
import subprocess
import time
from collections import OrderedDict
from math import log10

import lightning.pytorch as pl
import lmdb
import numpy as np
import option_func as options
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import util
from net.model import PromptIR
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rgb_to_grayscale
from tqdm import tqdm
from utils.image_io import save_image_tensor
from utils.val_utils import AverageMeter, compute_psnr_ssim

def calculate_psnr(img1, img2):
    """Calculate SSIM for numpy arrays"""
    # Convert tensor to numpy if needed
    if torch.is_tensor(img1):
        img1 = img1.squeeze(0).cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.squeeze(0).cpu().numpy()
    
    # Ensure correct shape (H, W, C)
    if img1.ndim == 3 and img1.shape[0] == 3:  # (C, H, W)
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] == 3:  # (C, H, W)
        img2 = np.transpose(img2, (1, 2, 0))
    
    return peak_signal_noise_ratio(img1, img2, data_range=1)

def calculate_ssim(img1, img2):
    """Calculate SSIM for numpy arrays"""
    # Convert tensor to numpy if needed
    if torch.is_tensor(img1):
        img1 = img1.squeeze(0).cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.squeeze(0).cpu().numpy()
    
    # Ensure correct shape (H, W, C)
    if img1.ndim == 3 and img1.shape[0] == 3:  # (C, H, W)
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] == 3:  # (C, H, W)
        img2 = np.transpose(img2, (1, 2, 0))
    
    return structural_similarity(img1, img2, multichannel=True, channel_axis=-1, data_range=1.0)


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]


class LQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        self.LR_size, self.GT_size = opt["LR_size"], opt["GT_size"]

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.LR_paths, self.LR_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":
            self.LR_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_LQ"]
            )  # LR list
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.LR_env = lmdb.open(
            self.opt["dataroot_LQ"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        GT_path, LR_path = None, None
        scale = self.opt["scale"] if self.opt["scale"] else 1
        GT_size = self.opt["patch_size"]
        LR_size = self.opt["patch_size"]

        # get GT image
        GT_path = self.GT_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # get LR image
        if self.LR_paths:  # LR exist
            LR_path = self.LR_paths[index]

            if self.opt["data_type"] == "lmdb":
                resolution = [int(s) for s in self.LR_sizes[index].split("_")]
            else:
                resolution = None
            img_LR = util.read_img(self.LR_env, LR_path, resolution)
 
        if self.opt["phase"] == "train":
            H, W, C = img_LR.shape
            assert LR_size == GT_size // scale, "GT size does not match LR size"

            # if img_GT.shape[0] != img_LR.shape[0]:
            #     img_GT = img_GT.transpose(1, 0, 2)
            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate

            img_LR, img_GT = util.augment(
                [img_LR, img_GT],
                self.opt["use_flip"],
                self.opt["use_rot"],
                mode=self.opt["mode"],
            )

            # img_GT = deg_util.usm_sharp(img_GT)

            if random.random() < 0.2:
                img_GT = util.channel_convert(img_GT.shape[2], 'gray', [img_GT])[0]
                img_LR = util.channel_convert(img_LR.shape[2], 'gray', [img_LR])[0]

        # change color space if necessary
        if self.opt["color"]:
            img_LR = util.channel_convert(img_LR.shape[2], self.opt["color"], [img_LR])[0]
            img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]

        lq4clip = util.clip_transform(img_LR)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        return {"LQ": img_LR, "GT": img_GT, "LQ_clip": lq4clip, "LQ_path": LR_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument("--opt", type=str, required=True, help="Path to options YMAL file.")
    parser.add_argument('--ckpt-name', type=str, default="model.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    opt = options.parse(testopt.opt, is_train=False)
    opt = options.dict_to_nonedict(opt)
    
    # get logger
    #### mkdir and logger
    util.mkdirs(
        (
            path
            for key, path in opt["path"].items()
            if not key == "experiments_root"
            and "pretrain_model" not in key
            and "resume" not in key
        )
    )

    os.system("rm ./result")
    os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"],
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(options.dict2str(opt))
    scale = opt['degradation']['scale']

    # Load model
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path = "ckpt/" + testopt.ckpt_name
    print("CKPT name : {}".format(ckpt_path))

    net  = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()
    
    def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
        phase = dataset_opt["phase"]
        if phase == "train":
            if opt["dist"]:
                world_size = torch.distributed.get_world_size()
                num_workers = dataset_opt["n_workers"]
                assert dataset_opt["batch_size"] % world_size == 0
                batch_size = dataset_opt["batch_size"] // world_size
                shuffle = False
            else:
                num_workers = dataset_opt["n_workers"] * len(opt["gpu_ids"])
                batch_size = dataset_opt["batch_size"]
                shuffle = True
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                sampler=sampler,
                drop_last=True,
                pin_memory=False,
            )
        else:
            return torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=(phase=="val")
            )


    #### Create test dataset and dataloader
    to_tensor = transforms.ToTensor()
    center_crop = transforms.CenterCrop([512,512])
    resize = transforms.Resize([512, 512], interpolation=transforms.InterpolationMode.BICUBIC)
    test_loaders = []
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = LQGTDataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        print(
            "Number of test images in [{:s}]: {:d}".format(
                dataset_opt["name"], len(test_set)
            )
        )
        test_loaders.append(test_loader)
    
    # Store test result
    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["time"] = []
    
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]  # path opt['']
        logger.info("\nTesting [{:s}]...".format(test_set_name))
        test_start_time = time.time()
        dataset_dir = os.path.join("/workspace/vunt/PromptIR/results", test_set_name)
        util.mkdir(dataset_dir)
        print(dataset_dir)
        
        # psnr = AverageMeter()
        # ssim = AverageMeter()
        psnr = []
        ssim = []

        start = time.time()
        for i, test_data in tqdm(enumerate(test_loader)):
            single_img_psnr = []
            single_img_ssim = []
            single_img_psnr_y = []
            single_img_ssim_y = []
            need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
            img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            #### input dataset_LQ
            LQ, GT = test_data["LQ"], test_data["GT"]
            
            # if LQ.shape[-1] > 512 or LQ.shape[-2] > 512:
            #     print(f"Resizing {img_name} from {LQ.shape[-2:]} to (512, 512)")
            #     resize = transforms.Resize((512, 512))
            #     LQ = resize(LQ)
            #     GT = resize(GT)
            
            # LQ = resize(LQ)
            # GT = resize(GT)
            
            LQ = center_crop(LQ)
            GT = center_crop(GT)
                
            with torch.no_grad():
                LQ = LQ.cuda()
                GT = GT.cuda()

                restored = net(LQ)
                            
            # LQ = LQ.squeeze().cpu().numpy()
            # GT = GT.squeeze().cpu().numpy()
            # restored = restored.squeeze().cpu().numpy()
                        
            # gt_img = GT / 255.0
            # sr_img = restored / 255.0

            # crop_border = opt["crop_border"] if opt["crop_border"] else scale
            # if crop_border == 0:
            #     cropped_sr_img = sr_img
            #     cropped_gt_img = gt_img
            # else:
            #     cropped_sr_img = sr_img[
            #         crop_border:-crop_border, crop_border:-crop_border
            #     ]
            #     cropped_gt_img = gt_img[
            #         crop_border:-crop_border, crop_border:-crop_border
            #     ]

            # psnr.append(util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255))
            # ssim.append(util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255))
            
            psnr.append(calculate_psnr(restored, GT))
            ssim.append(calculate_ssim(restored, GT))
            
            # temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, GT)

            # psnr.update(temp_psnr, N)
            # ssim.update(temp_ssim, N)
            
            # save image
            # util.save_img(restored, os.path.join(dataset_dir, img_name + '.png'))
            save_image_tensor(restored, os.path.join(dataset_dir, img_name + '.png'))
            
        avg_psnr = np.average(psnr)
        avg_ssim = np.average(ssim)
        # avg_psnr = psnr.avg
        # avg_ssim = ssim.avg
        eval_time = time.time() - start
        logger.info("Results for %s ----PSNR: %.2f, SSIM: %.4f in %.2f" % (test_set_name, avg_psnr, avg_ssim, eval_time))
        
        test_results["psnr"].append(avg_psnr)
        test_results["ssim"].append(avg_ssim)
        test_results["time"].append(eval_time)
        
    print(np.average(test_results["psnr"]), np.average(test_results["ssim"]), np.sum(test_results["time"]))
        
    logger.info("Avg PSNR: %.2f, Avg SSIM: %.4f in %.2f" % (np.average(test_results["psnr"]), np.average(test_results["ssim"]), np.sum(test_results["time"])))
        