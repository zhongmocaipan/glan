# import torch
# from data import Dataset
# from model import get_model
# from metric import cal_batch_psnr_ssim
# import pandas as pd
# from tqdm import tqdm
# import argparse
# from adapt import zsn2n, nbr2nbr
# import numpy as np
# from noise_generator import NoiseGenerator  # 引入噪声生成模型
# from torchvision.models import vgg16




# # 参数解析
# parser = argparse.ArgumentParser()
# parser.add_argument("--method", type=str, required=True, choices=["finetune", "lan"])
# parser.add_argument("--self_loss", type=str, required=True, choices=["nbr2nbr", "zsn2n"])
# parser.add_argument("--alpha", type=float, default=0.06, help="Noise intensity scaling factor")  # 新增参数
# args = parser.parse_args()


# # 加载模型和损失函数
# model_generator = get_model
# model = model_generator().cpu()


# if args.self_loss == "zsn2n":
#     loss_func = zsn2n.loss_func
# elif args.self_loss == "nbr2nbr":
#     loss_func = nbr2nbr.loss_func
# else:
#     raise NotImplementedError

# # 初始化噪声生成模型
# noise_generator = NoiseGenerator().cpu()



# # 设置模型参数的 requires_grad
# if args.method == "finetune":
#     for param in model.parameters():
#         param.requires_grad = True
# else:
#     for param in model.parameters():
#         param.requires_grad = False

# # 数据加载器
# dataloader = torch.utils.data.DataLoader(Dataset("dataset/lq", "dataset/gt"), batch_size=1, shuffle=False)

# # 学习率
# lr = 5e-4 if args.method == "lan" else 5e-6
# # LAN 模块
# class Lan(torch.nn.Module):
#     def __init__(self, shape):
#         super(Lan, self).__init__()
#         self.phi = torch.nn.Parameter(torch.zeros(shape), requires_grad=True)

#     def forward(self, x):
#         spatial_offset = torch.tanh(self.phi)  # 空间自适应偏移量
#         return x + spatial_offset

# # 日志记录
# logs_key = ["psnr", "ssim"]
# total_logs = {key: [] for key in logs_key}
# inner_loop = 21
# p_bar = tqdm(dataloader, ncols=100, desc=f"{args.method}_{args.self_loss}")

# # 噪声路径
# noise_path = "noise.pt"


# for lq, gt in p_bar:
#     lq = lq.cpu()
#     gt = gt.cpu()
#     lan = Lan(lq.shape).cpu() if args.method == "lan" else torch.nn.Identity()
#     model = model_generator().cpu()


#     for param in model.parameters():
#         param.requires_grad = args.method == "finetune"

#     params = list(lan.parameters()) if args.method == "lan" else list(model.parameters())
#     optimizer = torch.optim.Adam(params, lr=lr)
#     logs = {key: [] for key in logs_key}

#     for i in range(inner_loop):
#         optimizer.zero_grad()
#         # 生成噪声并调整输入图像
#         # 动态调整 alpha 参数
#         # alpha = args.alpha * (i / inner_loop)

#         generated_noise = args.alpha * noise_generator(lq.shape).cpu()  # 控制噪声强度
#         ###
#         gaussian_noise = torch.randn_like(lq) * 0.1 # 添加高斯噪声
#         combined_noise = generated_noise + gaussian_noise
#         ###
#         adapted_lq = lan(lq + generated_noise).cpu() #

#         with torch.no_grad():
#             pred = model(adapted_lq).clip(0, 1).cpu()
#         loss = loss_func(adapted_lq, model, i, inner_loop)
#         loss.backward()
#         optimizer.step()
#         psnr, ssim = cal_batch_psnr_ssim(pred, gt)
#         for key in logs_key:
#             logs[key].append(locals()[key])

#         # 打印每次 inner_loop 的 PSNR 和 SSIM 值
#         print(f"Loop {i + 1}/{inner_loop}, PSNR: {psnr[-1]:.6f}, SSIM: {ssim[-1]:.6f}")

#     for key in logs_key:
#         total_logs[key].extend(np.array(logs[key]).transpose())

#     p_bar.set_postfix(
#         PSNR=f"{np.array(total_logs['psnr']).mean(0)[0]:.2f}->{np.array(total_logs['psnr']).mean(0)[-1]:.2f}",
#         SSIM=f"{np.array(total_logs['ssim']).mean(0)[0]:.3f}->{np.array(total_logs['ssim']).mean(0)[-1]:.3f}"
#     )

#     # # 保存结果
#     # df_dict = {
#     #     "idx": [i for i in range(len(total_logs['psnr'])) for _ in range(inner_loop+1)],
#     #     "loop": [i for i in range(inner_loop+1)] * len(total_logs['psnr']),
#     # }
#     # for key in logs_key:
#     #     df_dict[key] = [value for value_list in total_logs[key] for value in value_list]
#     # df = pd.DataFrame(df_dict)
#     # df.to_csv(f"result_{args.method}_{args.self_loss}_GLAN.csv", index=False)
#     # print(df.groupby('loop').mean()[['psnr', 'ssim']])
# #python "D:\国防科技大学\大三\AIBD\肺分割论文\image denoise\代码\LAN-master\LAN-master\GLAN.py" --method lan --self_loss zsn2n

# #python "D:\国防科技大学\大三\AIBD\肺分割论文\image denoise\代码\LAN-master\LAN-master\mainfire.py" --method lan --self_loss zsn2n
import torch
from data import Dataset
from model import get_model
from metric import cal_batch_psnr_ssim
import pandas as pd
from tqdm import tqdm
import argparse
from adapt import zsn2n, nbr2nbr
import numpy as np
from noise_generator import NoiseGenerator  # 引入噪声生成模型
from torchvision.models import vgg16
import torch

import torch
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()


# 选择设备（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, required=True, choices=["finetune", "lan"])
parser.add_argument("--self_loss", type=str, required=True, choices=["nbr2nbr", "zsn2n"])
parser.add_argument("--alpha", type=float, default=0.06, help="Noise intensity scaling factor")  # 新增参数
args = parser.parse_args()

# 加载模型和损失函数
model_generator = get_model
model = model_generator().to(device)  # 迁移到 GPU

if args.self_loss == "zsn2n":
    loss_func = zsn2n.loss_func
elif args.self_loss == "nbr2nbr":
    loss_func = nbr2nbr.loss_func
else:
    raise NotImplementedError

# 初始化噪声生成模型
noise_generator = NoiseGenerator().to(device)  # 迁移到 GPU

# 设置模型参数的 requires_grad
if args.method == "finetune":
    for param in model.parameters():
        param.requires_grad = True
else:
    for param in model.parameters():
        param.requires_grad = False

# 数据加载器
dataloader = torch.utils.data.DataLoader(Dataset("dataset/lq", "dataset/gt"), batch_size=1, shuffle=False)

# 学习率
lr = 5e-4 if args.method == "lan" else 5e-6

# LAN 模块
class Lan(torch.nn.Module):
    def __init__(self, shape):
        super(Lan, self).__init__()
        self.phi = torch.nn.Parameter(torch.zeros(shape).to(device), requires_grad=True)  # 直接在 GPU 初始化

    def forward(self, x):
        spatial_offset = torch.tanh(self.phi)  # 空间自适应偏移量
        return x + spatial_offset

# 日志记录
logs_key = ["psnr", "ssim"]
total_logs = {key: [] for key in logs_key}
inner_loop = 21
p_bar = tqdm(dataloader, ncols=100, desc=f"{args.method}_{args.self_loss}")

# 噪声路径
noise_path = "noise.pt"

for lq, gt in p_bar:
    lq = lq.to(device)
    gt = gt.to(device)
    lan = Lan(lq.shape).to(device) if args.method == "lan" else torch.nn.Identity().to(device)
    model = model_generator().to(device)  # 确保模型在 GPU

    for param in model.parameters():
        param.requires_grad = args.method == "finetune"

    params = list(lan.parameters()) if args.method == "lan" else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    logs = {key: [] for key in logs_key}

    for i in range(inner_loop):
        optimizer.zero_grad()
        
        # 生成噪声并调整输入图像
        generated_noise = args.alpha * noise_generator(lq.shape).to(device)  # 控制噪声强度
        gaussian_noise = torch.randn_like(lq).to(device) * 0.1  # 添加高斯噪声
        combined_noise = generated_noise + gaussian_noise

        adapted_lq = lan(lq + generated_noise).to(device)  # 确保计算在 GPU

        with torch.no_grad():
            pred = model(adapted_lq).clip(0, 1).to(device)

        loss = loss_func(adapted_lq, model, i, inner_loop)
        loss.backward()
        optimizer.step()

        psnr, ssim = cal_batch_psnr_ssim(pred, gt)
        for key in logs_key:
            logs[key].append(locals()[key])

        # 打印每次 inner_loop 的 PSNR 和 SSIM 值
        print(f"Loop {i + 1}/{inner_loop}, PSNR: {psnr[-1]:.6f}, SSIM: {ssim[-1]:.6f}")

    for key in logs_key:
        total_logs[key].extend(np.array(logs[key]).transpose())

    p_bar.set_postfix(
        PSNR=f"{np.array(total_logs['psnr']).mean(0)[0]:.2f}->{np.array(total_logs['psnr']).mean(0)[-1]:.2f}",
        SSIM=f"{np.array(total_logs['ssim']).mean(0)[0]:.3f}->{np.array(total_logs['ssim']).mean(0)[-1]:.3f}"
    )
