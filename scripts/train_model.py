#!/usr/bin/env python
# coding: utf-8

import os,sys
import pdb

sys.path.insert(0,"..")
import os,sys,inspect
from glob import glob
from os.path import exists, join
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import pickle

import torch
import torchvision, torchvision.transforms
import skimage.transform
import sklearn, sklearn.model_selection

import random
import train_utils
import torchxrayvision as xrv


parser = argparse.ArgumentParser()
# parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('--name', type=str)
parser.add_argument('--output_dir', type=str, default="../outputs/")
parser.add_argument('--dataset', type=str, default="chex")
parser.add_argument('--dataset_dir', type=str, default="/lotterlab/datasets/")
parser.add_argument('--model', type=str, default="resnet50")
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--num_epochs', type=int, default=400, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--threads', type=int, default=4, help='')
parser.add_argument('--taskweights', type=bool, default=True, help='')
parser.add_argument('--no_taskweights', dest='taskweights', action='store_false')
parser.add_argument('--featurereg', type=bool, default=False, help='')
parser.add_argument('--weightreg', type=bool, default=False, help='')
parser.add_argument('--data_aug', type=bool, default=False, help='')
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
parser.add_argument('--label_concat', type=bool, default=False, help='')
parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
parser.add_argument('--labelunion', type=bool, default=False, help='')
parser.add_argument('--im_size', type=int, default=512, help='')
parser.add_argument('--data_aug_window_width_min', type=int, default=None, help='')
parser.add_argument('--data_aug_max_resize', type=int, default=None, help='')
parser.add_argument('--gpu', '-g', default='0', help='which gpu to use', required=True)
parser.add_argument('--label_type', default='pathology')
parser.add_argument('--fixed_splits', action='store_true')
parser.add_argument('--all_views', action='store_true')
parser.add_argument('--use_scheduler', action='store_true')
parser.add_argument('--class_balance', action='store_true')
parser.add_argument('--multifactorial_class_balance', action='store_true')
parser.add_argument('--imagenet_pretrained', action='store_true')
parser.add_argument('--use_no_finding', default=False, action='store_true')

cfg = parser.parse_args()
cfg.output_dir = os.path.join(cfg.output_dir, cfg.name + '/')
print(cfg)

if not os.path.exists(cfg.output_dir):
    os.mkdir(cfg.output_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

data_aug = None
#if cfg.data_aug:
    # data_aug = torchvision.transforms.Compose([
    #     xrv.datasets.ToPILImage(),
    #     torchvision.transforms.RandomAffine(cfg.data_aug_rot,
    #                                         translate=(cfg.data_aug_trans, cfg.data_aug_trans),
    #                                         scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
    #     torchvision.transforms.ToTensor()
    # ])
    # print(data_aug)
if cfg.data_aug_max_resize:
    transforms = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.RandomZoom(cfg.data_aug_max_resize, cfg.im_size)])
    transforms_val = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(cfg.im_size)])
else:
    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(cfg.im_size)])
    transforms_val = transforms

if 'race' in cfg.label_type:
    labels_to_use = ['Mapped_Race']
    n_classes = int(cfg.label_type[-1])
    if n_classes == 2:
        labels_to_use += ['Black', 'White']
    else:
        labels_to_use += ['Asian', 'Black', 'White']
else:
    labels_to_use = None

#use_class_balancing = not cfg.taskweights # if taskweights aren't use, do equal sampling
if cfg.multifactorial_class_balance:
    use_class_balancing = 'multifactorial'
else:
    use_class_balancing = cfg.class_balance

if use_class_balancing == 'multifactorial':
    assert cfg.dataset in ['chex', 'mimic_ch']
    data_tag = 'cxp' if cfg.dataset == 'chex' else 'mimic'
    print('data_tag', data_tag)
    class_balancing_labels_df = pd.read_csv(f'../data/{data_tag}_confounder_df.csv', index_col=0)
    with open(f'../data/{data_tag}_confounder_proportions-ABW.pkl', 'rb') as f:
        all_props = pickle.load(f)

    class_balancing_props = []
    #class_balancing_props.append(('Mapped_Race', ['Asian', 'Black', 'White']))
    for confounder in ['Mapped_Race', 'Age_Bin', 'Sex']:
        class_balancing_props.append((confounder, all_props[confounder]))

    paths_to_use = ["Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
            'No Finding']
    path_props = {}
    for pathology in paths_to_use:
        path_props[pathology] = all_props[pathology]
    class_balancing_props.append(('pathology', path_props))
else:
    class_balancing_props = None
    class_balancing_labels_df = None

datas = []
datas_names = []
if "nih" in cfg.dataset:
    dataset = xrv.datasets.NIH_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-NIH", 
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("nih")
if "pc" in cfg.dataset:
    dataset = xrv.datasets.PC_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-PC", 
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("pc")
if "chex" in cfg.dataset:
    if cfg.all_views:
        views = 'all'
    else:
        views = ['PA', 'AP']
    print('views', views)
    if cfg.fixed_splits:
        csvpath = '/lotterlab/lotterb/project_data/bias_interpretability/cxp_cv_splits/version_0/train.csv'

        valid_dataset = xrv.datasets.CheX_Dataset(
            imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
            csvpath=csvpath.replace('train', 'val'),
            transform=transforms_val, data_aug=data_aug, unique_patients=False,
            min_window_width=None, views=views,
            labels_to_use=labels_to_use, use_class_balancing=use_class_balancing,
            use_no_finding=cfg.use_no_finding,
            class_balancing_labels_df=class_balancing_labels_df,
            class_balancing_proportions=class_balancing_props)
    else:
        csvpath = cfg.dataset_dir + "/CheXpert-v1.0-small/train.csv"
        valid_dataset = None

    dataset = xrv.datasets.CheX_Dataset(
        imgpath=cfg.dataset_dir + "/CheXpert-v1.0-small",
        csvpath=csvpath,
        transform=transforms, data_aug=data_aug, unique_patients=False,
        min_window_width=cfg.data_aug_window_width_min, views=views,
        labels_to_use=labels_to_use, use_class_balancing=use_class_balancing,
        use_no_finding=cfg.use_no_finding,
        class_balancing_labels_df=class_balancing_labels_df,
        class_balancing_proportions=class_balancing_props)

    datas.append(dataset)
    datas_names.append("chex")
if "google" in cfg.dataset:
    dataset = xrv.datasets.NIH_Google_Dataset(
        imgpath=cfg.dataset_dir + "/images-512-NIH",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("google")
if "mimic_ch" in cfg.dataset:
    imgpath = '/lotterlab/datasets/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/physionet.org/files/mimic-cxr-jpg/2.0.0/files_small/'
    if cfg.all_views:
        views = 'all'
    else:
        views = ['PA', 'AP']
    print('views', views)
    if cfg.fixed_splits:
        csvpath = '/lotterlab/lotterb/project_data/bias_interpretability/mimic_cv_splits/version_0/cxp-labels_train.csv'
        metacsvpath = csvpath.replace('cxp-labels', 'meta')

        valid_dataset = xrv.datasets.MIMIC_Dataset(
            imgpath=imgpath,
            csvpath=csvpath.replace('train', 'val'),
            metacsvpath=metacsvpath.replace('train', 'val'),
            transform=transforms_val, data_aug=data_aug, unique_patients=False,
            min_window_width=None, views=views,
            labels_to_use=labels_to_use, use_class_balancing=use_class_balancing,
            use_no_finding=cfg.use_no_finding,
            class_balancing_labels_df=class_balancing_labels_df,
            class_balancing_proportions=class_balancing_props)
    else:
        csvpath = None # not set up yet
        metacsvpath = None
        valid_dataset = None

    dataset = xrv.datasets.MIMIC_Dataset(
        imgpath=imgpath,
        csvpath=csvpath,
        metacsvpath=metacsvpath,
        transform=transforms, data_aug=data_aug, unique_patients=False,
        min_window_width=cfg.data_aug_window_width_min, views=views,
        labels_to_use=labels_to_use, use_class_balancing=use_class_balancing,
        use_no_finding=cfg.use_no_finding,
        class_balancing_labels_df=class_balancing_labels_df,
        class_balancing_proportions=class_balancing_props)

    datas.append(dataset)
    datas_names.append("mimic_ch")
if "openi" in cfg.dataset:
    dataset = xrv.datasets.Openi_Dataset(
        imgpath=cfg.dataset_dir + "/OpenI/images/",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("openi")
if "rsna" in cfg.dataset:
    dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
        imgpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
        transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA","AP"])
    datas.append(dataset)
    datas_names.append("rsna")
if "siim" in cfg.dataset:
    dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
        imgpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/dicom-images-train",
        csvpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/train-rle.csv",
        transform=transforms, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("siim")
if "vin" in cfg.dataset:
    dataset = xrv.datasets.VinBrain_Dataset(
        imgpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train",
        csvpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train.csv",
        transform=transform, data_aug=data_aug)
    datas.append(dataset)
    datas_names.append("vin")


print("datas_names", datas_names)

if not labels_to_use:
    if cfg.labelunion:
        newlabels = set()
        for d in datas:
            newlabels = newlabels.union(d.pathologies)
        newlabels.remove("Support Devices")
        print(list(newlabels))
        for d in datas:
            xrv.datasets.relabel_dataset(list(newlabels), d)
    else:
        if not cfg.use_no_finding:
            for d in datas:
                xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d)
            if valid_dataset is not None:
                xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, valid_dataset)

#cut out training sets
train_datas = []
test_datas = []
for i, dataset in enumerate(datas):
    if not cfg.fixed_splits:

        # give patientid if not exist
        if "patientid" not in dataset.csv:
            dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]

        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)

        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
        train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
        test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)

        np.save(cfg.output_dir + 'dataset_{}_train_inds.npy'.format(i), train_inds)
        np.save(cfg.output_dir + 'dataset_{}_test_inds.npy'.format(i), test_inds)

        #disable data augs
        #test_dataset.data_aug = None  # raises error
    
        train_datas.append(train_dataset)
        test_datas.append(test_dataset)

    else:
        train_datas.append(dataset)
        test_datas.append([])
    
if len(datas) == 0:
    raise Exception("no dataset")
elif len(datas) == 1:
    train_dataset = train_datas[0]
    test_dataset = test_datas[0]
else:
    print("merge datasets")
    train_dataset = xrv.datasets.Merge_Dataset(train_datas)
    test_dataset = xrv.datasets.Merge_Dataset(test_datas)


# Setting the seed
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.cuda:
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("train_dataset.labels.shape", train_dataset.labels.shape)
#print("test_dataset.labels.shape", test_dataset.labels.shape)
print("train_dataset",train_dataset)
#print("test_dataset",test_dataset)
    
# create models
if labels_to_use is None:
    n_classes = train_dataset.labels.shape[1]
    use_softmax = False
else:
    use_softmax = True

if "densenet" in cfg.model:
    if cfg.imagenet_pretrained:
        weights = 'imagenet'
    else:
        weights = None
    model = xrv.models.DenseNet(num_classes=n_classes, in_channels=1, weights=weights,
                                **xrv.models.get_densenet_params(cfg.model)) 
elif "resnet101" in cfg.model:
    model = torchvision.models.resnet101(num_classes=n_classes, pretrained=False)
    #patch for single channel
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
elif "resnet50" in cfg.model:
    model = torchvision.models.resnet50(num_classes=n_classes, pretrained=False)
    #patch for single channel
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
elif "shufflenet_v2_x2_0" in cfg.model:
    model = torchvision.models.shufflenet_v2_x2_0(num_classes=n_classes, pretrained=False)
    #patch for single channel
    model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
elif "squeezenet1_1" in cfg.model:
    model = torchvision.models.squeezenet1_1(num_classes=n_classes, pretrained=False)
    #patch for single channel
    model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
else:
    raise Exception("no model")


train_utils.train(model, train_dataset, cfg, valid_dataset, use_softmax)


print("Done")
# test_loader = torch.utils.data.DataLoader(test_dataset,
#                                            batch_size=cfg.batch_size,
#                                            shuffle=cfg.shuffle,
#                                            num_workers=0, pin_memory=False)






