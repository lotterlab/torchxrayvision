#!/usr/bin/env python
# coding: utf-8

"""
GAN-based Windowing for X-ray Images
Two discriminators:
  1. Real vs Windowed image discriminator
  2. Dataset/scanner source discriminator
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import pickle
from glob import glob
from os.path import exists, join

import torch
import torch.nn as nn
import torchvision
import sklearn.model_selection

# Add parent directory to path for torchxrayvision
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchxrayvision as xrv
import torchxrayvision.train_utils_gan as train_utils_gan

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--output_dir', type=str, default="/lotterlab/lotterb/project_data/autopreprocess/gan_windowing_models/")
parser.add_argument('--dataset', type=str, default="chex")
parser.add_argument('--dataset_dir', type=str, default="/lotterlab/datasets/")

# Model architecture
parser.add_argument('--generator_model', type=str, default="resnet50", help='Hypernetwork backbone')
parser.add_argument('--disc_model', type=str, default="resnet18", help='Discriminator architecture')
parser.add_argument('--window_nbins', type=int, default=8, help='Number of bins for spline windowing')

# Training parameters
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--lr_gen', type=float, default=0.0002, help='Generator learning rate')
parser.add_argument('--lr_disc', type=float, default=0.0002, help='Discriminator learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
parser.add_argument('--threads', type=int, default=4)

# Loss weights
parser.add_argument('--lambda_adv', type=float, default=1.0, help='Weight for adversarial loss')
parser.add_argument('--lambda_dataset', type=float, default=1.0, help='Weight for dataset discriminator loss')
parser.add_argument('--lambda_reg', type=float, default=0.2, help='Weight for regularization loss')
parser.add_argument('--reg_type', type=str, default='l2', choices=['l2', 'l1', 'perceptual'], 
                    help='Type of regularization loss')

# Dataset parameters
parser.add_argument('--im_size', type=int, default=512)
parser.add_argument('--gpu', '-g', default='0', required=True)
parser.add_argument('--fixed_splits', action='store_true')
parser.add_argument('--all_views', action='store_true')

# GAN-specific parameters
parser.add_argument('--n_critic', type=int, default=5, help='Number of discriminator updates per generator update')
parser.add_argument('--use_wgan', action='store_true', help='Use WGAN-GP instead of standard GAN')
parser.add_argument('--gp_lambda', type=float, default=10.0, help='Gradient penalty lambda for WGAN-GP')

# Save/load
parser.add_argument('--save_freq', type=int, default=5, help='Save model every N epochs')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

cfg = parser.parse_args()
cfg.output_dir = os.path.join(cfg.output_dir, cfg.name + '/')
print(cfg)

if not os.path.exists(cfg.output_dir):
    os.makedirs(cfg.output_dir, exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

# Set random seeds
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data transforms
transforms = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(), 
    xrv.datasets.XRayResizer(cfg.im_size)
])
transforms_val = transforms

# Load datasets
print("Loading datasets...")
print(f"Requested dataset(s): {cfg.dataset}")

# Parse dataset string (e.g., "chex+mimic_ch" -> ["chex", "mimic_ch"])
dataset_list = [d.strip() for d in cfg.dataset.split('+')]
print(f"Parsed datasets: {dataset_list}")

datas = []
datas_names = []
valid_datasets = []  # Store validation datasets separately

# Load NIH
if "nih" in dataset_list:
    print("Loading NIH dataset...")
    dataset = xrv.datasets.NIH_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-NIH", 
        transform=transforms, data_aug=None, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("nih")
    valid_datasets.append(None)
    
# Load PadChest
if "pc" in dataset_list:
    print("Loading PadChest dataset...")
    dataset = xrv.datasets.PC_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-PC", 
        transform=transforms, data_aug=None, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("pc")
    valid_datasets.append(None)
    
# Load CheXpert
if "chex" in dataset_list:
    print("Loading CheXpert dataset...")
    views = 'all' if cfg.all_views else ['PA', 'AP']
    print(f'  views: {views}')
    
    if cfg.fixed_splits:
        csvpath = '/lotterlab/lotterb/project_data/bias_interpretability/cxp_cv_splits/version_0/train.csv'
        print(f'  Using fixed splits: {csvpath}')
        valid_dataset_chex = xrv.datasets.CheX_Dataset(
            imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
            csvpath=csvpath.replace('train', 'val'),
            transform=transforms_val, data_aug=None, unique_patients=False,
            min_window_width=None, views=views)
    else:
        csvpath = cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv"
        valid_dataset_chex = None

    dataset = xrv.datasets.CheX_Dataset(
        imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
        csvpath=csvpath,
        transform=transforms, data_aug=None, unique_patients=False,
        min_window_width=None, views=views)
    
    print(f'  Loaded {len(dataset)} training samples')
    datas.append(dataset)
    datas_names.append("chex")
    valid_datasets.append(valid_dataset_chex)

# Load MIMIC-CXR
if "mimic_ch" in dataset_list or "mimic" in dataset_list:
    print("Loading MIMIC-CXR dataset...")
    imgpath = '/lotterlab/datasets/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/physionet.org/files/mimic-cxr-jpg/2.0.0/files_small/'
    views = 'all' if cfg.all_views else ['PA', 'AP']
    print(f'  views: {views}')
    
    if cfg.fixed_splits:
        csvpath = '/lotterlab/lotterb/project_data/bias_interpretability/mimic_cv_splits/version_0/cxp-labels_train.csv'
        metacsvpath = csvpath.replace('cxp-labels', 'meta')
        print(f'  Using fixed splits: {csvpath}')
        
        valid_dataset_mimic = xrv.datasets.MIMIC_Dataset(
            imgpath=imgpath,
            csvpath=csvpath.replace('train', 'val'),
            metacsvpath=metacsvpath.replace('train', 'val'),
            transform=transforms_val, data_aug=None, unique_patients=False,
            min_window_width=None, views=views)
    else:
        csvpath = None
        metacsvpath = None
        valid_dataset_mimic = None

    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath=imgpath,
        csvpath=csvpath,
        metacsvpath=metacsvpath,
        transform=transforms, data_aug=None, unique_patients=False,
        min_window_width=None, views=views)
    
    print(f'  Loaded {len(dataset)} training samples')
    datas.append(dataset)
    datas_names.append("mimic_ch")
    valid_datasets.append(valid_dataset_mimic)

print(f"\n=== Dataset Summary ===")
print(f"Loaded {len(datas)} dataset(s): {datas_names}")
for i, (name, data) in enumerate(zip(datas_names, datas)):
    print(f"  {i}: {name} - {len(data)} samples")
print("=" * 40 + "\n")

# Combine validation datasets if multiple exist
valid_dataset = None
if any(v is not None for v in valid_datasets):
    valid_list = [v for v in valid_datasets if v is not None]
    if len(valid_list) == 1:
        valid_dataset = valid_list[0]
    elif len(valid_list) > 1:
        print("Merging validation datasets")
        valid_dataset = xrv.datasets.Merge_Dataset(valid_list)

# Split datasets into train/test
train_datas = []
test_datas = []

for i, dataset in enumerate(datas):
    if not cfg.fixed_splits:
        # Add patientid if not exist
        if "patientid" not in dataset.csv:
            dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, j) for j in range(len(dataset))]

        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8, test_size=0.2, random_state=cfg.seed)
        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
        
        train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
        test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)
        
        np.save(cfg.output_dir + f'dataset_{i}_train_inds.npy', train_inds)
        np.save(cfg.output_dir + f'dataset_{i}_test_inds.npy', test_inds)
        
        train_datas.append(train_dataset)
        test_datas.append(test_dataset)
    else:
        train_datas.append(dataset)
        test_datas.append([])

if len(datas) == 0:
    raise Exception("No dataset loaded")
elif len(datas) == 1:
    train_dataset = train_datas[0]
    test_dataset = test_datas[0] if test_datas[0] else None
else:
    print("Merging datasets")
    train_dataset = xrv.datasets.Merge_Dataset(train_datas)
    test_dataset = None

# Add dataset labels to each sample for the dataset discriminator
# This assumes datasets are merged in order
num_datasets = len(datas)
dataset_labels = []
for i, d in enumerate(train_datas):
    dataset_labels.extend([i] * len(d))
dataset_labels = torch.tensor(dataset_labels)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Number of datasets: {num_datasets}")

# Create models
print("Creating models...")
from torchxrayvision.models_gan import Generator, RealFakeDiscriminator, DatasetDiscriminator

# Generator (hypernetwork outputting spline parameters)
generator = Generator(
    backbone=cfg.generator_model,
    window_nbins=cfg.window_nbins
)

# Discriminator 1: Real vs Windowed
disc_real_fake = RealFakeDiscriminator(
    architecture=cfg.disc_model
)

# Discriminator 2: Dataset source
disc_dataset = DatasetDiscriminator(
    architecture=cfg.disc_model,
    num_datasets=num_datasets
)

if cfg.cuda:
    generator = generator.cuda()
    disc_real_fake = disc_real_fake.cuda()
    disc_dataset = disc_dataset.cuda()
    dataset_labels = dataset_labels.cuda()

print("Generator:", generator)
print("Discriminator (Real/Fake):", disc_real_fake)
print("Discriminator (Dataset):", disc_dataset)

# Train
print("Starting training...")
train_utils_gan.train_gan(
    generator=generator,
    disc_real_fake=disc_real_fake,
    disc_dataset=disc_dataset,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    dataset_labels=dataset_labels,
    cfg=cfg
)

print("Training complete!")