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

torch.cuda.empty_cache()
x = torch.randn(64, 3, 256, 256).cuda()
with torch.no_grad():
    z     = model.encode(x)
    x_rec = model.decode(z)
mem = torch.cuda.memory_allocated()/1e9
print(f'batch=64 chunked decode FITS — mem: {mem:.2f}GB')
print(f'x_rec shape: {tuple(x_rec.shape)}')
print('DECODE FIX CONFIRMED')
