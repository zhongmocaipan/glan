import importlib
from pathlib import Path
import torch

def get_model():
    path = 'Restormer.basicsr.models.archs.restormer_arch'
    module_spec = importlib.util.spec_from_file_location(path, str(Path().joinpath(*path.split('.')))+'.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # model = module.Restormer(LayerNorm_type = 'BiasFree').cuda()
    model = module.Restormer(LayerNorm_type='BiasFree').cpu()
    checkpoint = torch.load("D:/国防科技大学/大三/AIBD/肺分割论文/image denoise/代码/LAN-master/LAN-master/checkpoint/real_denoising.pth",map_location=torch.device('cpu'), weights_only=True)["params"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model