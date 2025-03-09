import importlib
from pathlib import Path
import torch

def get_model():
    path = 'Restormer.basicsr.models.archs.restormer_arch'
    module_spec = importlib.util.spec_from_file_location(path, str(Path().joinpath(*path.split('.')))+'.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.Restormer(LayerNorm_type = 'BiasFree').cpu()
    # model = module.Restormer(LayerNorm_type='BiasFree').cpu()
    checkpoint = torch.load("./checkpoint/real_denoising.pth",map_location=torch.device('cpu'), weights_only=True)["params"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model