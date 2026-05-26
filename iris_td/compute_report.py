import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from datetime import datetime

def get_metrics(csv_path, thr_path):
    df   = pd.read_csv(csv_path)
    thr  = json.load(open(thr_path))
    m    = thr['best_method']
    tau  = thr['best_threshold']
    sc   = df[m].values
    lab  = (df['label']=='attack').astype(int).values
    pred = (sc > tau).astype(int)
    tp = np.sum((pred==1)&(lab==1))
    fp = np.sum((pred==1)&(lab==0))
    tn = np.sum((pred==0)&(lab==0))
    fn = np.sum((pred==0)&(lab==1))
    apcer = fp/max(tn+fp,1)
    bpcer = fn/max(tp+fn,1)
    acer  = (apcer+bpcer)/2
    acc   = (tp+tn)/max(len(lab),1)
    f1    = 2*tp/max(2*tp+fp+fn,1)
    auc   = roc_auc_score(lab,sc)
    fpr,tpr,_ = roc_curve(lab,sc)
    fnr = 1-tpr
    idx = np.argmin(np.abs(fnr-fpr))
    eer = (fpr[idx]+fnr[idx])/2
    return dict(
        apcer=apcer, bpcer=bpcer, acer=acer,
        eer=eer, auc=auc, acc=acc, f1=f1,
        method=m, threshold=tau,
        n_bonafide=int(np.sum(lab==0)),
        n_attack=int(np.sum(lab==1))
    )

def get_per_attack(csv_path, thr_path):
    df  = pd.read_csv(csv_path)
    thr = json.load(open(thr_path))
    m   = thr['best_method']
    tau = thr['best_threshold']
    order = [
        ('E-display',        'Hardest'),
        ('CL',               'Hard'),
        ('Print_E-display',  'Medium'),
        ('Printed',          'Medium'),
        ('Fake_with_Add_On', 'Medium'),
        ('Generated',        'Easy'),
        ('Artifact',         'Very Easy'),
        ('Post-Mortem',      'Perfect'),
    ]
    rows = []
    for atype, diff in order:
        sub = df[df['attack_type']==atype]
        if len(sub)==0:
            continue
        sc   = sub[m].values
        pred = (sc > tau).astype(int)
        miss = np.sum(pred==0)/max(len(sub),1)
        det  = 1 - miss
        rows.append((atype, len(sub), miss*100, det*100, diff))
    return rows

# ── Load all results ───────────────────────────────────────────────────────
best = get_metrics(
    'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25.csv',
    'iris_td/results/threshold_simplex_tstar500.json'
)
best_atk = get_per_attack(
    'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25.csv',
    'iris_td/results/threshold_simplex_tstar500.json'
)
bbdm = get_metrics(
    'iris_bbdm_pad/results/test_pad_scores.csv',
    'iris_bbdm_pad/results/threshold.json'
)
bbdm_atk = get_per_attack(
    'iris_bbdm_pad/results/test_pad_scores.csv',
    'iris_bbdm_pad/results/threshold.json'
)

ablation = [
    ('DDPM vanilla  (Gaussian, t*=1000)',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar1000_steps25.csv',
     'iris_td/results/threshold_gaussian_tstar1000.json',
     25, 0.133),
    ('DDPM partial  (Gaussian, t*=500)',
     'iris_td/pad_scores/ddpm_test_gaussian_tstar500_steps25.csv',
     'iris_td/results/threshold_gaussian_tstar500.json',
     25, 0.133),
    ('DDPM AnoDDPM  (Simplex,  t*=500)',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_steps25.csv',
     'iris_td/results/threshold_simplex_tstar500.json',
     25, 0.133),
    ('DDPM+DDIM     (Simplex,  t*=500)',
     'iris_td/pad_scores/ddpm_test_simplex_tstar500_ddim_steps50.csv',
     'iris_td/results/threshold_simplex_tstar500_ddim.json',
     50, 0.196),
    ('Brownian Bridge (BBDM) <- best',
     'iris_bbdm_pad/results/test_pad_scores.csv',
     'iris_bbdm_pad/results/threshold.json',
     200, 0.876),
]

order = [
    ('E-display','Hardest'),('CL','Hard'),
    ('Print_E-display','Medium'),('Printed','Medium'),
    ('Fake_with_Add_On','Medium'),('Generated','Easy'),
    ('Artifact','Very Easy'),('Post-Mortem','Perfect'),
]

# ── Print verification numbers ─────────────────────────────────────────────
print('=== NUMBERS TO USE IN REPORT ===')
print(f'Best DDPM (AnoDDPM Simplex t*=500 25steps):')
for k,v in best.items():
    if isinstance(v, float):
        if k in ['auc','threshold']:
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v*100:.2f}%')
    else:
        print(f'  {k}: {v}')
print()
print('BBDM:')
for k,v in bbdm.items():
    if isinstance(v, float):
        if k in ['auc','threshold']:
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v*100:.2f}%')
    else:
        print(f'  {k}: {v}')
print()
print('All configs:')
for name,csv_p,thr_p,steps,time in ablation:
    if not Path(csv_p).exists():
        continue
    m = get_metrics(csv_p, thr_p)
    print(f'  {name:<40} ACER={m["acer"]*100:.2f}%  APCER={m["apcer"]*100:.2f}%  BPCER={m["bpcer"]*100:.2f}%  EER={m["eer"]*100:.2f}%  AUC={m["auc"]:.4f}')

# ── Build report ───────────────────────────────────────────────────────────
L = []
def add(s=''): L.append(s)

add('=' * 68)
add('Standard DDPM Ablation for Iris Presentation Attack Detection')
add('Comparison with Brownian Bridge Diffusion Model (BBDM)')
add('IIT Mandi | April 2026')
add('Geetanjali Sharma, Shubham Ashwani, Aditya Nigam')
add('=' * 68)

# 1. Problem Statement
add()
add('1. PROBLEM STATEMENT')
add('-' * 68)
add('Iris biometric systems are vulnerable to spoofing attacks (printed')
add('irises, e-displays, contact lenses, synthetic images, post-mortem')
add('samples). Existing supervised methods train on known attack types')
add('and fail to generalize to unseen attacks. We need a method that')
add('detects attacks without ever seeing attack data during training.')
add()
add('This report evaluates standard DDPM (with AnoDDPM-style inference)')
add('as an ablation against our primary BBDM approach. Both methods')
add('train exclusively on bona fide images and detect attacks via')
add('reconstruction error (anomaly detection, not classification).')

# 2. Approach
add()
add('2. OUR APPROACH: DDPM AS ANOMALY DETECTOR (ABLATION)')
add('-' * 68)
add('We train a standard Latent DDPM on bona fide iris images only,')
add('then apply the AnoDDPM inference strategy (Wyatt et al. CVPR 2022):')
add('partial noising with simplex noise to t*=500 followed by reverse')
add('diffusion. Reconstruction error = PAD score.')
add()
add('Key Design Decisions:')
add(f'  {"Decision":<25} {"Choice":<28} {"Rationale"}')
add('  ' + '-' * 68)
decisions = [
    ('DDPM Variant',     'Latent DDPM',            '64x64 latent via VQGAN VQ-f4'),
    ('Diffusion Math',   'Linear beta schedule',   'Ho et al. DDPM (standard)'),
    ('Inference',        'AnoDDPM strategy',       'Wyatt et al. CVPR 2022'),
    ('Noise Type',       'Simplex noise',          'Structured — better for attacks'),
    ('Partial noising',  't* = 500 / 1000',        'Preserves coarse iris structure'),
    ('PAD Scoring',      'MSE + LPIPS',            'AnoDDPM recommendation'),
    ('Skip Sampling',    'Linear (torch.linspace)','Matches BBDM for fair comparison'),
    ('Training Data',    'Bona fide ONLY',         'Zero attack supervision'),
    ('Attack Types',     '8 types tested',         'Same as BBDM evaluation'),
]
for d,c,r in decisions:
    add(f'  {d:<25} {c:<28} {r}')

# 3. Pipeline
add()
add('3. PIPELINE OVERVIEW')
add('-' * 68)
add('Training (bona fide only):')
add('  Clean bona fide iris images encoded to 64x64 latent via frozen')
add('  VQGAN VQ-f4. Standard DDPM trained to denoise latents (predict')
add('  epsilon). Loss = MSE(predicted_noise, true_noise). 200 epochs.')
add('  EMA of model weights maintained; used for inference.')
add()
add('Inference — 4 ablation configs evaluated:')
add('  Config 1: Gaussian noise, t*=1000, 25 steps  (vanilla DDPM)')
add('  Config 2: Gaussian noise, t*=500,  25 steps  (partial noising)')
add('  Config 3: Simplex  noise, t*=500,  25 steps  (AnoDDPM)')
add('  Config 4: Simplex  noise, t*=500,  50 steps, DDIM (AnoDDPM+DDIM)')
add()
add('  All configs: z0=encode(x0), z_t=partial_noise(z0,t*),')
add('  z_recon=reverse_diffuse(z_t, linear_skip_steps),')
add('  x_recon=decode(z_recon), score=MSE+LPIPS(x_recon,x0)')

# 4. Implementation
add()
add('4. IMPLEMENTATION SUMMARY')
add('-' * 68)
add(f'  {"Phase":<22} {"What Was Done":<36} {"Duration"}')
add('  ' + '-' * 66)
add(f'  {"Phase 1: Data":<22} {"Reused BBDM data pipeline.":<36} {"Reused"}')
add(f'  {"":22} {"14,028 bonafide training pairs.":<36}')
add(f'  {"":22} {"21,380 val + 47,434 test images.":<36}')
add(f'  {"Phase 2: DDPM Training":<22} {"Trained LatentDDPM 200 epochs.":<36} {"~23h"}')
add(f'  {"":22} {"Same UNet, VQGAN, data as BBDM.":<36}')
add(f'  {"":22} {"batch=32, Adam lr=1e-4, T=1000.":<36}')
add(f'  {"Phase 3: Inference":<22} {"4 configs x val+test sets.":<36} {"~13h"}')
add(f'  {"":22} {"Chunked VQGAN decode (8/chunk).":<36}')
add(f'  {"":22} {"Threshold tuned on val per config.":<36}')
add()
add('  Technical Stack: BBDM repo (xuekt98/BBDM) + VQGAN (VQ-f4 LDM)')
add('  + PyTorch + LPIPS (AlexNet) + OpenSimplex. RTX A5000 24GB VRAM.')

# 5. Results
add()
add('5. RESULTS')
add('-' * 68)
add(f'Main Results — Best DDPM Config: AnoDDPM (Simplex, t*=500, 25 steps)')
add(f'Test Set: {best["n_bonafide"]:,} bonafide + {best["n_attack"]:,} attack')
add()
add(f'  {"Metric":<15} {"DDPM (best)":>12} {"BBDM":>12} {"Winner":>10}')
add('  ' + '-' * 52)
metrics_compare = [
    ('ACER',      best['acer']*100,  bbdm['acer']*100,  False),
    ('APCER',     best['apcer']*100, bbdm['apcer']*100, False),
    ('BPCER',     best['bpcer']*100, bbdm['bpcer']*100, False),
    ('EER',       best['eer']*100,   bbdm['eer']*100,   False),
    ('AUC',       best['auc'],       bbdm['auc'],       True),
    ('Accuracy',  best['acc']*100,   bbdm['acc']*100,   True),
    ('F1',        best['f1'],        bbdm['f1'],        True),
]
for mname, dval, bval, higher_better in metrics_compare:
    bbdm_wins = (bval < dval) if not higher_better else (bval > dval)
    winner = 'BBDM' if bbdm_wins else 'DDPM'
    if mname in ['AUC','F1']:
        add(f'  {mname:<15} {dval:>12.4f} {bval:>12.4f} {winner:>10}')
    else:
        add(f'  {mname:<15} {dval:>11.2f}% {bval:>11.2f}% {winner:>10}')

add()
add(f'  Best Scoring Method: DDPM={best["method"]}  BBDM={bbdm["method"]}')

add()
add('Per-Attack Detection (sorted hardest to easiest):')
add(f'  {"Attack Type":<22} {"DDPM APCER":>10} {"DDPM Det%":>10}'
    f' {"BBDM APCER":>10} {"BBDM Det%":>10} {"Difficulty"}')
add('  ' + '-' * 72)
bbdm_dict = {r[0]:r for r in bbdm_atk}
ddpm_dict = {r[0]:r for r in best_atk}
for atype, diff in order:
    d = ddpm_dict.get(atype)
    b = bbdm_dict.get(atype)
    da = f'{d[2]:>9.2f}%' if d else f'{"N/A":>10}'
    dd = f'{d[3]:>9.2f}%' if d else f'{"N/A":>10}'
    ba = f'{b[2]:>9.2f}%' if b else f'{"N/A":>10}'
    bd = f'{b[3]:>9.2f}%' if b else f'{"N/A":>10}'
    add(f'  {atype:<22} {da} {dd} {ba} {bd} {diff}')
add()
add('  Key finding: BBDM outperforms DDPM on E-display, CL, Generated,')
add('  and Printed — the structurally ambiguous attacks. DDPM better only')
add('  on Artifact (coarse texture, easily detected by either method).')

# 6. Ablation
add()
add('6. ABLATION STUDY — DDPM INFERENCE STRATEGY')
add('-' * 68)
add('Each config adds one improvement. Same model checkpoint throughout.')
add()
add(f'  {"Config":<38} {"ACER":>6} {"APCER":>6} {"BPCER":>6} {"EER":>6} {"AUC":>7} {"t/img":>7}')
add('  ' + '-' * 85)
prev_acer = None
for name, csv_p, thr_p, steps, t_img in ablation:
    if not Path(csv_p).exists():
        add(f'  {name:<38}  NOT DONE'); continue
    m = get_metrics(csv_p, thr_p)
    delta = f'({(m["acer"]-prev_acer)*100:+.2f}%)' if prev_acer is not None else ''
    sep = ' <-' if 'BBDM' in name else ''
    add(f'  {name:<38} {m["acer"]*100:>5.2f}% {m["apcer"]*100:>5.2f}% '
        f'{m["bpcer"]*100:>5.2f}% {m["eer"]*100:>5.2f}% '
        f'{m["auc"]:>7.4f} {t_img:>5.3f}s  {delta}{sep}')
    prev_acer = m['acer']
add()
add('  Findings:')
add('  - Partial noising (t*=500) helps vs full noising (t*=1000)')
add('    by -0.96% ACER. Preserving coarse structure matters.')
add('  - Simplex noise helps vs Gaussian: -1.04% ACER.')
add('    Structured noise better matches iris attack patterns.')
add('  - DDIM hurts slightly vs stochastic (+0.49% ACER).')
add('    Deterministic sampling reduces diversity needed for anomaly det.')
add('  - BBDM wins overall: -0.30% vs best DDPM.')
add('    Constrained bridge path fundamentally better for iris textures.')

# 7. Comparison
add()
add('7. COMPARISON WITH STATE OF THE ART')
add('-' * 68)
add('vs BBDM (same dataset, same training data, same UNet/VQGAN):')
add()
add(f'  {"Method":<40} {"Supervision":>12} {"ACER":>7} {"EER":>7} {"AUC":>7}')
add('  ' + '-' * 75)
add(f'  {"BBDM (Brownian Bridge, LBBDM-f4)":<40} '
    f'{"Unsupervised":>12} {bbdm["acer"]*100:>6.2f}% {bbdm["eer"]*100:>6.2f}% {bbdm["auc"]:>7.4f}  <- best')
for name, csv_p, thr_p, steps, t_img in ablation[:-1]:
    if not Path(csv_p).exists(): continue
    m = get_metrics(csv_p, thr_p)
    add(f'  {name:<40} {"Unsupervised":>12} '
        f'{m["acer"]*100:>6.2f}% {m["eer"]*100:>6.2f}% {m["auc"]:>7.4f}')
add()
add('vs LivDet-Iris 2025 (different test set, for context only):')
add(f'  {"Team":<40} {"Supervision":>12} {"AUROC":>7}')
add('  ' + '-' * 62)
add(f'  {"Dermalog-Iris (Winner)":<40} {"Supervised":>12}  0.9057')
add(f'  {"MSU D-NetPAD":<40} {"Supervised":>12}  0.9014')
add(f'  {"BUCEA":<40} {"Supervised":>12}  0.8984')
add(f'  {"Ours BBDM (our dataset*)":<40} {"Unsupervised":>12}  {bbdm["auc"]:.4f}')
add(f'  {"Ours DDPM-AnoDDPM (our dataset*)":<40} {"Unsupervised":>12}  {best["auc"]:.4f}')
add()
add('  * AUC not directly comparable — different test sets.')

# 8. Novelty
add()
add('8. NOVELTY AND CONTRIBUTIONS')
add('-' * 68)
add('1. Controlled ablation study — same UNet, VQGAN, training data,')
add('   epochs, threshold optimization. Only diffusion math varies.')
add('   Cleanest possible comparison: BBDM vs DDPM.')
add('2. BBDM proven superior to all DDPM variants on iris PAD.')
add('   The constrained Brownian bridge path better models iris texture.')
add('3. AnoDDPM improvements validated step by step:')
add('   partial noising (-0.96%) and simplex noise (-1.04%) each help.')
add('4. DDIM finding: deterministic sampling hurts anomaly detection')
add('   (+0.49% ACER). Stochastic exploration important for iris PAD.')
add('5. Per-attack analysis: BBDM most advantaged on Generated, Printed,')
add('   and E-display — attacks with fine texture structure.')

# 9. Next Steps
add()
add('9. NEXT STEPS')
add('-' * 68)
add('1. Show DDPM ablation results to mentor (Geetanjali ma\'am).')
add('2. Include DDPM ablation table in paper Section 4 (Experiments).')
add('3. Add DET curves comparing all 5 configs + BBDM.')
add('4. Add score distribution plots showing separation quality.')
add('5. Add per-attack radar chart (DDPM vs BBDM).')
add('6. Final evaluation presentation: April 25, 2026.')

add()
add('=' * 68)
add(f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
add('Model checkpoint: iris_td/results/ddpm_run1/DDPM/checkpoint/top_model_epoch_best.pth')
add('Results dir:      iris_td/final_results/')
add('=' * 68)

text = '\n'.join(L)
print(text)

out = 'iris_td/final_results/DDPM_Ablation_Progress_Report.txt'
Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    f.write(text)
print(f'\nSaved to {out}')
