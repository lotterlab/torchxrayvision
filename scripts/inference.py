"""
run inference for pathology and score models
combine the infernece outputs with the ground truth in one df

combined from analysis.py and results.py (/lotterlab/users/jfernando/project_1/scripts)

date created: 2023/09/01
updates: 2023/11/17 - moved merging with ground truth df to xray_generalization/merge_predictions_with_gt_labels.py
"""

import numpy as np
import os
import pandas as pd
import skimage
import sys
import torch
import torch.nn.functional as F
import torchvision as tv
import tqdm


from functools import partial
from pycrumbs import tracked # add pycrumbs to track inference runs (the records are saved with the dataset csv files)

sys.path.append('/lotterlab/users/khoebel/xray_generalization/scripts/torchxrayvision/')
import torchxrayvision as xrv

sys.path.append('/lotterlab/users/jfernando/project_1/scripts/')
from data_utils import create_cxp_view_column, get_mimic_jpg_path, apply_window
from constants import PROJECT_DIR, CXP_JPG_DIR,  CXP_LABELS # KVH: MODEL_SAVE_DIR, overwriting some of the constants at other points in this script



def load_model(model_save_dir, model_name, checkpoint_name, num_classes):
    model = xrv.models.DenseNet(num_classes=num_classes, in_channels=1,
                                **xrv.models.get_densenet_params('densenet')).cuda()

    model_pref = '{}-densenet'.format('chex' if 'cxp' in model_name else 'mimic_ch') # auto added by xrv
    weights_path = os.path.join(model_save_dir, model_name, '{}-{}-{}.pt'.format(model_pref, model_name, checkpoint_name))
    model.load_state_dict(torch.load(weights_path).state_dict())

    model.eval()
    return model


# KVH: copied and modified this from /lotterlab/users/jfernando/project_1/scripts/data_utils
def load_split_metadf(dataset, split, only_good_files=True):
    # base_cv_dir = PROJECT_DIR # + '{}_cv_splits/'.format(dataset) # KVH - commented out
    if dataset == 'cxp':
        fname = os.path.join(PROJECT_DIR, split + '.csv')
    elif dataset == 'mmc':
        fname = os.path.join(PROJECT_DIR, 'meta_' + split + '.csv')

    df = pd.read_csv(fname)
    if dataset == 'mmc' and only_good_files:
        good_dicoms = np.load(
            '/lotterlab/datasets/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.0.0/good_image_dicoms.npy')
        df = df[df.dicom_id.isin(good_dicoms)].copy()
    return df


def xrv_preprocess(image_path, final_resize=224, window_width=None, init_resize=None, start_at_top=True, additional_transforms = []):
    # KVH: added option for additional transforms (to allow the use of custom transforms)
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = skimage.io.imread(image_path)

    if init_resize is None:
        init_resize = final_resize

    if window_width and window_width != 256:
        img = apply_window(img, 256. / 2, window_width, y_min=0, y_max=255)

    img = xrv.datasets.normalize(img, maxval=255, reshape=True)

    transforms = [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(init_resize)]
    transforms.extend(additional_transforms) # add custom transforms 
    transform = tv.transforms.Compose(transforms)
    img = transform(img)

    if init_resize != final_resize:
        diff = int((init_resize - final_resize) / 2)
        if start_at_top:
            img = img[:, :final_resize, diff:(diff + final_resize)]
        else:
            img = img[:, diff:(diff + final_resize), diff:(diff + final_resize)]

    return img



def run_predictions(model_name, 
                    checkpoint_name, 
                    dataset, 
                    model_save_dir, # KVH: added to avoid setting global variables
                    split='test',
                    prediction_mode='pathology',
                    window_width = None, # KVH
                    init_resize=None, # KVH
                    start_at_top=None, # KVH,
                    additional_transforms=[],
                    savedir = None, # KVH
                    score_model_target='' # modifier used if running a model trained on a different target, eg. the PNX score model on pneumonia
                    ):
    # KVH: added window_width, init_resize, start_at_top as additional parameters 
    # (required within the function but not provided)

    print(prediction_mode)
    # KVH modified to accommodate both pathology and higher-score predictions 
    if prediction_mode == 'pathology':
        labels = CXP_LABELS
        df = load_split_metadf(dataset, split)
        
    elif prediction_mode == 'higher_score':
        # KVH: load score labels
        # KVH: needs to be changed once we have a 35% score model
        labels = ['CXP','MMC']
        if dataset == 'cxp': # KVH: changed labels to dataset 
            print('load cxp dataframe')
            df = pd.read_csv(os.path.join(PROJECT_DIR,'{}.csv'.format(split)))
            # df = pd.read_csv('/lotterlab/users/jfernando/project_1/data/cxp_cv_splits/pneumothorax/{}.csv'.format(split))
        elif dataset == 'mmc':
            print('load mmc dataframe')
            df = pd.read_csv(os.path.join(PROJECT_DIR, 'meta_{}.csv'.format(split)))
            # df = pd.read_csv('/lotterlab/users/jfernando/project_1/data/mimic_cv_splits/pneumothorax/meta_{}.csv'.format(split))
    
    model = load_model(model_save_dir, model_name, checkpoint_name, len(labels))

    
    print(df.head())
    if dataset == 'cxp':
        df['file_path'] = [CXP_JPG_DIR + v for v in df.Path.values]
        df = create_cxp_view_column(df)
    else:
        df['file_path'] = df.apply(partial(get_mimic_jpg_path, small=True), axis=1)
        df['view'] = df.ViewPosition

    im_proc_fxn = partial(xrv_preprocess, 
                          window_width=window_width, 
                          init_resize=init_resize, 
                          start_at_top=start_at_top, 
                          additional_transforms=additional_transforms)

    pred_data = []
    with torch.no_grad():
        # df = df.head(20)
        for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            try:
                x = torch.from_numpy(im_proc_fxn(row['file_path'])).unsqueeze(0).cuda()
                if prediction_mode == 'higher_score':
                    preds = F.softmax(model(x), dim=1).cpu().squeeze().numpy()
                elif prediction_mode == 'pathology':
                    preds = torch.sigmoid(model(x)).cpu().squeeze().numpy()
                pred_data.append([row['file_path'], row['view']] + preds.tolist())
            except:
                pred_data.append([row['file_path'], row['view']] + [np.nan]*len(labels))   

            # KVH: not sure we wheather this does anything and we need this - 
            # commented out for now 
            '''if not os.path.exists(row['file_path']):
                continue
            '''

    pred_df = pd.DataFrame(pred_data, columns=['Path', 'View'] + ['Pred_' + r for r in labels])
    out_dir = os.path.join(PROJECT_DIR, 'prediction_dfs', score_model_target+model_name + '-' + checkpoint_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    out_name = dataset + '-' + split +'.csv'
    pred_df.to_csv(os.path.join(out_dir, out_name))




@tracked(directory_parameter='PROJECT_DIR')
def inference_dataset(PROJECT_DIR:str, 
                      model_dir_dict:dict, 
                      model_name_dict:dict,
                      splits:list,
                      prediction_mode:str,
                      checkpoint_name = 'best',
                      additional_transforms=[],
                      score_model_target=''
                      ):
    

    for model_key in model_dir_dict.keys():
        model_save_dir = model_dir_dict[model_key]
        print(model_save_dir)

        for model_name in model_name_dict[model_key]:
            print(model_name)
        
            for split in splits:
                print('running prediction on', dataset, 'using the ', model_key, prediction_mode, 'model')
                run_predictions(model_name=model_name, 
                                checkpoint_name = checkpoint_name, 
                                dataset=dataset, 
                                split=split,
                                model_save_dir=model_save_dir,
                                prediction_mode=prediction_mode,
                                window_width = None, # KVH
                                init_resize=None, # KVH
                                start_at_top=None, # KVH
                                additional_transforms=additional_transforms,# KVH
                                savedir = None, 
                                score_model_target=score_model_target # KVH
                                )
    



if __name__ == '__main__':


    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    prediction_mode = 'higher_score' # 'higher_score', 'pathology'
    
    # splits = ['val', 'test', 'train_score']

    # 35% pathology model 
    # splits = ['val', 'test', 'train']
    '''project_dir_dict = {'mmc':"/lotterlab/users/khoebel/xray_generalization/data/splits/mmc/0.35/pathology",
                        'cxp': "/lotterlab/users/khoebel/xray_generalization/data/splits/cxp/0.35/pathology"}
    
    model_dir_dict = {'mmc': "/lotterlab/users/khoebel/xray_generalization/models/mmc/0.35/pathology",
                      'cxp': '/lotterlab/users/khoebel/xray_generalization/models/cxp/0.35/pathology'
                      }
    
    model_name_dict = {'mmc': [# 'mmc_densenet_pretrained_0.35', 
                               'mmc_densenet_pretrained_0.35_seed_1',
                               'mmc_densenet_pretrained_0.35_seed_2'],
                       'cxp': ['cxp_densenet_pretrained_0.35', 
                                'cxp_densenet_pretrained_0.35_seed_1',
                                'cxp_densenet_pretrained_0.35_seed_2']
                   }'''
    
    
    # 70% pathology model 
    '''
    splits = ['val', 'test', 'train']
    project_dir_dict = {'mmc':"/lotterlab/users/khoebel/xray_generalization/data/splits/mimic/0.7/pathology",
                        'cxp': "/lotterlab/users/khoebel/xray_generalization/data/splits/cxp/0.7/pathology"
                        }
    
    model_dir_dict = {'mmc': "/lotterlab/users/jfernando/project_1/repos/torchxrayvision/outputs",
                      'cxp': "/lotterlab/users/jfernando/project_1/repos/torchxrayvision/outputs"
                      }
    
    model_name_dict = {'mmc': ['mimic_densenet_pretrained_v2', 
                               'mimic_densenet_pretrained_v3',
                               'mimic_densenet_pretrained_v4'],
                       'cxp': ['cxp_densenet_pretrained_v2', 
                               'cxp_densenet_pretrained_v3',
                               'cxp_densenet_pretrained_v4']
                   }'''
    
    '''# .35 score model
    splits = ['val', 'test', 'train_score']
    model_dir_dict = {'mmc':"/lotterlab/users/khoebel/xray_generalization/models/mmc/0.35/pneumothorax",
                      'cxp': "/lotterlab/users/khoebel/xray_generalization/models/cxp/0.35/pneumothorax"
                      }
    

    model_name_dict = {'mmc': ['mmc_score_0.35_seed_1_DS32'], # list of all names of models for inference
                       'cxp': ['cxp_score_0.35_seed_1_DS32']
                       }
    

    project_dir_dict = {'mmc':"/lotterlab/users/khoebel/xray_generalization/data/splits/mmc/0.35/pneumothorax",
                        'cxp': "/lotterlab/users/khoebel/xray_generalization/data/splits/cxp/0.35/pneumothorax"
                        }'''
    

    # .7 score model
    # splits = ['val', 'test']
    splits = ['test']
    model_dir_dict = {'mmc':"/lotterlab/users/khoebel/xray_generalization/models/mmc/0.7/pneumonia",
                      'cxp': "/lotterlab/users/khoebel/xray_generalization/models/cxp/0.7/pneumonia"
                      }
    

    model_name_dict = {'mmc': ['mmc_score_0.7_seed_1'], # list of all names of models for inference
                       'cxp': ['cxp_score_0.7_seed_1']
                       }
    

    project_dir_dict = {'mmc':"/lotterlab/users/khoebel/xray_generalization/data/splits/mmc/0.7/effusion",
                        'cxp': "/lotterlab/users/khoebel/xray_generalization/data/splits/cxp/0.7/effusion"
                        }
    
    additional_transforms = []
    # additional_transforms = [xrv.datasets.DownSample(patch_size =32)]

    # loop through project directories (i.e., datasets to run prediction on)
    for dataset in ['cxp', 'mmc']:
        PROJECT_DIR = project_dir_dict[dataset]
        print(PROJECT_DIR)
        inference_dataset(PROJECT_DIR=PROJECT_DIR,
                          model_dir_dict=model_dir_dict, 
                          model_name_dict=model_name_dict,
                          splits=splits,
                          prediction_mode=prediction_mode,
                          checkpoint_name='best',
                          additional_transforms=additional_transforms, 
                          score_model_target='pneumonia_'
                          )
    