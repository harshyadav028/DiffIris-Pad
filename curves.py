#SET THRESHOLDS MANUALLY BASED ON VALIDATION RESULTS BEFORE RUNNING THIS SCRIPT

import torch
import torch.nn as nn
import torch.optim as optim
from open_set_training import training,get_data_loaders
import argparse
from open_set_dataset_loader import datasetLoader
from models import MaxViTModel, ResNetModel , DINOResNetModel, DenseNetModel, DinoV2ViTModel, EnsembleModel2, ViTModel
from tqdm import tqdm
import multiprocessing
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=32)
parser.add_argument('-csvPath', required=False, default= 'Images/combined_dataset.csv',type=str)
parser.add_argument('-datasetPath', required=False, default= 'Images/',type=str)
parser.add_argument('-use_amp', default= False,type=bool)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

def calculate_eer(fpr, tpr):
    fnr = 1 - tpr
    eer_threshold = np.nanargmin(np.absolute((fnr - fpr)))
    eer = np.mean([fpr[eer_threshold], fnr[eer_threshold]])
    return eer * 100

def plot_det_curve(predict, real, attack, name, all_curves_data):
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(real, predict)

    # Calculate BPCER and APCER
    bpcer = fpr
    apcer = 1 - tpr

    # Calculate EER
    eer = calculate_eer(fpr, tpr)

    # Store curve data for combined plot
    all_curves_data[name] = {
        'apcer': apcer,
        'bpcer': bpcer,
        'attack': attack
    }

    # Create individual DET curve
    plt.figure(figsize=(10, 8))
    plt.plot(apcer * 100, bpcer * 100, 'b-', label=f'{name}')

    plt.xlabel('APCER (%)')
    plt.ylabel('BPCER (%)')
    plt.title(f'DET Curve for {attack} Attack')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='lower right')

    # Save individual plot
    os.makedirs(f'Open_Set/{name}/{attack}/Results', exist_ok=True)
    plt.savefig(f'Open_Set/{name}/{attack}/Results/det_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save operating points with EER
    operating_points = {
        'APCER1': None,
        'APCER10': None,
        'APCER20': None
    }

    for i in range(len(apcer)):
        if apcer[i] <= 0.01 and operating_points['APCER1'] is None:
            operating_points['APCER1'] = bpcer[i]
        if apcer[i] <= 0.1 and operating_points['APCER10'] is None:
            operating_points['APCER10'] = bpcer[i]
        if apcer[i] <= 0.2 and operating_points['APCER20'] is None:
            operating_points['APCER20'] = bpcer[i]

    with open(f'Open_Set/{name}/{attack}/Results/operating_points.txt', 'w') as f:
        f.write(f'BPCER @ APCER = 1%: {operating_points["APCER1"]*100:.2f}%\n')
        f.write(f'BPCER @ APCER = 10%: {operating_points["APCER10"]*100:.2f}%\n')
        f.write(f'BPCER @ APCER = 20%: {operating_points["APCER20"]*100:.2f}%\n')
        f.write(f'EER: {eer:.2f}%\n')

MODEL_COLORS = {
    'ResNet50':          ('blue',       '--'),
    'ViT-B':             ('red',        '--'),
    'MaxViT':            ('green',      '--'),
    'DINOv1+ResNet50':   ('cyan',       '--'),
    'DINOv2+ViTs14':     ('magenta',    '--'),
}

def plot_combined_curves(all_curves_data, attack):
    plt.figure(figsize=(10, 8))

    for name, data in all_curves_data.items():
        if data['attack'] != attack:
            continue
        color, linestyle = MODEL_COLORS.get(name, ('gray', '--'))
        plt.plot(
            data['apcer'] * 100,
            data['bpcer'] * 100,
            linestyle=linestyle,
            color=color,
            label=name,
            linewidth=2.5
        )

    plt.xlabel('APCER (%)', fontsize=16)
    plt.ylabel('BPCER (%)', fontsize=16)
    plt.title(f'Combined DET Curves for {attack} Attack', fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.legend(
        loc='upper right',
        fontsize=14,
    )

    plt.tight_layout()

    os.makedirs('Open_Set/curves', exist_ok=True)
    plt.savefig(
        f'Open_Set/curves/combined_det_curve_{attack}.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

all_curves_data = {}

attacks_list = ['Artifact','CL','E-display','Fake with Add On','Generated','PostMortem','Print and E-display','Printed']
for attack in attacks_list:
    for name in ['ResNet50','vit_base_patch16_224','MaxViTModel','DinoResNet50','DinoV2']:
        print(attack,name)
        dataset = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i={'Live':0,'Spoof':1},attack_type=attack)
        print(f'len of test = {len(dataset)}')
        print('\n\n')
        num_workers = min(32, multiprocessing.cpu_count())
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=num_workers, pin_memory=True)

        if name == 'ResNet50':
            model = ResNetModel()
            graphName='ResNet50'
        elif name == 'vit_base_patch16_224':
            model = ViTModel(variant='vit_base_patch16_224')
            graphName='ViT-B'
        elif name == 'MaxViTModel':
            model = MaxViTModel()
            graphName='MaxViT'
        elif name == 'DinoV2':
            model = DinoV2ViTModel()
            graphName='DINOv2+ViTs14'
        elif name == 'DinoResNet50':
            model = DINOResNetModel()
            graphName='DINOv1+ResNet50'

        model_dict = torch.load(f'Open_Set/{name}/{attack}/Logs/{name}_best.pth', map_location=device)['state_dict']
        model.load_state_dict(model_dict)
        model = model.to(device)

        testPredScore = []
        testTrueLabel = []
        torch.cuda.empty_cache()
        print(f'Testing {name} with {attack}...')
        model.eval()
        with torch.no_grad():
            all_outputs, all_labels = [], []
            for data, labels,imageName, _ in tqdm(test_loader, desc="Testing", leave=False):
                data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                with torch.amp.autocast('cuda',enabled=args.use_amp):
                    outputs = model(data)

                all_outputs.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())

            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            predict = np.array(all_outputs)
            predict = softmax(predict, axis=1)
            if(len(predict.shape)==2):
                predict =  predict[:, 1]
            elif(len(predict.shape)==3):
                predict = predict[:, :, 1]
            real = np.array(all_labels)

        plot_det_curve(predict, real, attack, graphName, all_curves_data)

    plot_combined_curves(all_curves_data, attack)
