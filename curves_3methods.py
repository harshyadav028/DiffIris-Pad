"""
Combined DET curves for f-AnoGAN vs AnoDDPM+DDIM+Simplex vs Diff-IrisPAD.

Unlike curves.py (which re-runs classifier inference), this reads cached
per-sample score CSVs directly, so no model loading / GPU inference is
needed:
  - f-AnoGAN:              iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv  (vit_score)
  - AnoDDPM+DDIM+Simplex:  iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50_vit.csv  (vit_score)
  - Diff-IrisPAD:          iris_bbdm_pad/results/test_pad_scores.csv  (recon_score)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

ROOT = os.path.dirname(os.path.abspath(__file__))

ATTACKS_LIST = [
    'Artifact', 'CL', 'E-display', 'Fake with Add On',
    'Generated', 'PostMortem', 'Print and E-display', 'Printed'
]

# Maps curves.py-style attack names -> attack_type values used in each CSV
ANODDPM_ATTACK_MAP = {
    'Fake with Add On': 'Fake_with_Add_On',
    'Print and E-display': 'Print_E-display',
}

METHODS = {
    'f-AnoGAN': {
        'csv': os.path.join(ROOT, 'iris_anogan/results/fanogan_run1/pad_scores/test_scores.csv'),
        'score_col': 'vit_score',
        'attack_map': {},
    },
    'AnoDDPM DDIM Simplex': {
        'csv': os.path.join(ROOT, 'iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50_vit.csv'),
        'score_col': 'vit_score',
        'attack_map': ANODDPM_ATTACK_MAP,
    },
    'Diff-IrisPAD': {
        'csv': os.path.join(ROOT, 'iris_bbdm_pad/results/test_pad_scores.csv'),
        'score_col': 'recon_score',
        'attack_map': {},
    },
}

MODEL_COLORS = {
    'f-AnoGAN':              ('blue',   '--'),
    'AnoDDPM DDIM Simplex':  ('purple', '--'),
    'Diff-IrisPAD':          ('orange', '--'),
}


def calculate_eer(fpr, tpr):
    fnr = 1 - tpr
    eer_threshold = np.nanargmin(np.absolute((fnr - fpr)))
    eer = np.mean([fpr[eer_threshold], fnr[eer_threshold]])
    return eer * 100


def plot_det_curve(predict, real, attack, name, all_curves_data):
    fpr, tpr, thresholds = roc_curve(real, predict)

    bpcer = fpr
    apcer = 1 - tpr

    eer = calculate_eer(fpr, tpr)

    all_curves_data[name] = {
        'apcer': apcer,
        'bpcer': bpcer,
        'attack': attack
    }

    plt.figure(figsize=(10, 8))
    plt.plot(apcer * 100, bpcer * 100, 'b-', label=f'{name}')
    plt.xlabel('APCER (%)')
    plt.ylabel('BPCER (%)')
    plt.title(f'DET Curve for {attack} Attack')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='lower right')

    os.makedirs(f'Open_Set/{name}/{attack}/Results', exist_ok=True)
    plt.savefig(f'Open_Set/{name}/{attack}/Results/det_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    operating_points = {'APCER1': None, 'APCER10': None, 'APCER20': None}
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
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()

    os.makedirs('Open_Set/curves', exist_ok=True)
    plt.savefig(
        f'Open_Set/curves/combined_det_curve_{attack}.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def main():
    dfs = {}
    for name, cfg in METHODS.items():
        df = pd.read_csv(cfg['csv'])
        df['real'] = (df['label'] == 'attack').astype(int)
        dfs[name] = df

    all_curves_data = {}
    for attack in ATTACKS_LIST:
        for name, cfg in METHODS.items():
            df = dfs[name]
            csv_attack = cfg['attack_map'].get(attack, attack)
            mask = (df['attack_type'] == csv_attack) | (df['label'] == 'bonafide')
            sub = df[mask]

            predict = sub[cfg['score_col']].values
            real = sub['real'].values

            print(f'{attack} | {name} | n={len(sub)} (bonafide={ (real==0).sum() }, attack={(real==1).sum()})')
            plot_det_curve(predict, real, attack, name, all_curves_data)

        plot_combined_curves(all_curves_data, attack)


if __name__ == '__main__':
    main()
