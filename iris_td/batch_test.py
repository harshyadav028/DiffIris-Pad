import sys, torch, yaml
sys.path.insert(0, 'iris_td/models')
sys.path.insert(0, 'BBDM')
from ddpm_model import LatentDDPM

cfg = yaml.safe_load(open('iris_td/configs/ddpm_iris.yaml'))
model = LatentDDPM(
    cfg['model']['unet_config'],
    cfg['model']['vqgan_config'],
    cfg['model']['ddpm_config'],
).cuda()
model.eval()
print(f'Model loaded: {torch.cuda.memory_allocated()/1e9:.2f}GB')

for batch in [48, 32, 24, 16, 8]:
    torch.cuda.empty_cache()
    try:
        x = torch.randn(batch, 3, 256, 256).cuda()
        with torch.no_grad():
            z      = model.encode(x)
            t      = torch.randint(0, 1000, (batch,)).cuda()
            z_t, _ = model.q_sample(z, t)
            timesteps = torch.linspace(999, 1, 25, dtype=torch.long).cuda()
            for t_val in timesteps:
                t_b = t_val.expand(batch)
                z_t = model.p_sample(z_t, t_b)
            x_rec = model.decode(z_t)
        mem = torch.cuda.memory_allocated()/1e9
        print(f'batch={batch} FITS — peak mem: {mem:.2f}GB')
        print(f'USE batch_size={batch} for Configs 1-3')
        break
    except torch.cuda.OutOfMemoryError:
        print(f'batch={batch} OOM')
        torch.cuda.empty_cache()
