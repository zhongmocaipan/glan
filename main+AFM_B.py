# import torch
# from data import Dataset
# from model import get_model
# from metric import cal_batch_psnr_ssim
# import pandas as pd
# from tqdm import tqdm
# import argparse
# from adapt import zsn2n, nbr2nbr
# import numpy as np
# from AFM_B import AFM_B

# parser = argparse.ArgumentParser()
# parser.add_argument("--method", type=str, required=True, choices=["finetune", "lan"])
# parser.add_argument("--self_loss", type=str, required=True, choices=["nbr2nbr", "zsn2n"])
# args = parser.parse_args()
# if args.self_loss == "zsn2n":
#     loss_func = zsn2n.loss_func
# elif args.self_loss == "nbr2nbr":
#     loss_func = nbr2nbr.loss_func
# else:
#     raise NotImplementedError

# model_generator = get_model
# model = model_generator()

# # for param in model.parameters():
# #     param.requires_grad = args.method == "finetune"
# #     param.requires_grad = True
# # print("trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
# # 设置模型参数的 requires_grad
# if args.method == "finetune":
#     for param in model.parameters():
#         param.requires_grad = True
# else:
#     for param in model.parameters():
#         param.requires_grad = False
# print("trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# dataloader = torch.utils.data.DataLoader(Dataset("dataset/lq", "dataset/gt"), batch_size=1, shuffle=False)


# lr = 5e-4 if args.method == "lan" else 5e-6

# class LanWithAFM(torch.nn.Module):
#     def __init__(self, shape, channels, fq_bound=1.0):
#         super(LanWithAFM, self).__init__()
#         self.phi = torch.nn.Parameter(torch.zeros(shape), requires_grad=True)
#         self.afm = AFM_B(fq_bound=fq_bound)  # 集成AFM-B

#     def forward(self, x):
#         # 假设x是噪声图像，clean是干净图像（可以通过其他方式获取）
#         clean = torch.zeros_like(x)  # 示例：使用零图像作为干净图像
#         _, _, fq_mask = self.afm(clean, x)  # 生成掩码

#         adapted_x = x * fq_mask + torch.tanh(self.phi)  # 应用掩码
#         return adapted_x


# logs_key = ["psnr", "ssim"]
# total_logs = {key: [] for key in logs_key}
# inner_loop = 20 #20
# p_bar = tqdm(dataloader, ncols=100, desc=f"{args.method}_{args.self_loss}")

# for lq, gt in p_bar:
#     # lq = lq.cuda()
#     # gt = gt.cuda()
#     lq = lq.cpu()
#     gt = gt.cpu()
#     # lan = Lan(lq.shape).cuda() if args.method == "lan" else torch.nn.Identity()
#     lan = LanWithAFM(lq.shape).cpu() if args.method == "lan" else torch.nn.Identity()
#     tmp_batch_size = lq.shape[0]
#     model = model_generator().cpu()

#     for param in model.parameters():
#         # print(f"requires_grad: {param.requires_grad}")
#         param.requires_grad = args.method == "finetune"

#     params = list(lan.parameters()) if args.method == "lan" else list(model.parameters())
#     optimizer = torch.optim.Adam(params, lr=lr)
#     logs = {key: [] for key in logs_key}

#     for i in range(inner_loop):
#         optimizer.zero_grad()
#         adapted_lq = lan(lq).cpu()
#         with torch.no_grad():
#             pred = model(adapted_lq).clip(0, 1).cpu()
#         loss = loss_func(adapted_lq, model, i, inner_loop)
#         loss.backward()
#         optimizer.step()
#         psnr, ssim = cal_batch_psnr_ssim(pred, gt)
#         for key in logs_key:
#             logs[key].append(locals()[key])
#     else:
#         with torch.no_grad():
#             adapted_lq = lan(lq).cpu()
#             pred = model(adapted_lq).clip(0, 1).cpu()
#             psnr, ssim = cal_batch_psnr_ssim(pred, gt)
#         for key in logs_key:
#             logs[key].append(locals()[key])

#     for key in logs_key:
#         total_logs[key].extend(np.array(logs[key]).transpose())
#     p_bar.set_postfix(
#         PSNR=f"{np.array(total_logs['psnr']).mean(0)[0]:.2f}->{np.array(total_logs['psnr']).mean(0)[-1]:.2f}",
#         SSIM=f"{np.array(total_logs['ssim']).mean(0)[0]:.3f}->{np.array(total_logs['ssim']).mean(0)[-1]:.3f}"
#     )


#     df_dict = {
#         "idx": [i for i in range(len(total_logs['psnr'])) for _ in range(inner_loop+1)],
#         "loop": [i for i in range(inner_loop+1)] * len(total_logs['psnr']),
#     }
#     for key in logs_key:
#         df_dict[key] = [value for value_list in total_logs[key] for value in value_list]
#     df = pd.DataFrame(df_dict)
#     df.to_csv(f"result_{args.method}_{args.self_loss}_origion.csv", index=False)
#     print(df.groupby('loop').mean()[['psnr', 'ssim']])

# #python "D:\国防科技大学\大三\AIBD\肺分割论文\image denoise\代码\LAN-master\LAN-master\main.py" --method lan --self_loss zsn2n
# import torch
# from data import Dataset
# from model import get_model
# from metric import cal_batch_psnr_ssim
# import pandas as pd
# from tqdm import tqdm
# import argparse
# from adapt import zsn2n, nbr2nbr
# import numpy as np
# from AFM_B import AFM_B

# # 解析命令行参数
# parser = argparse.ArgumentParser()
# parser.add_argument("--method", type=str, required=True, choices=["finetune", "lan"])
# parser.add_argument("--self_loss", type=str, required=True, choices=["nbr2nbr", "zsn2n"])
# args = parser.parse_args()

# # 选择损失函数
# if args.self_loss == "zsn2n":
#     loss_func = zsn2n.loss_func
# elif args.self_loss == "nbr2nbr":
#     loss_func = nbr2nbr.loss_func
# else:
#     raise NotImplementedError

# # 获取模型
# model_generator = get_model
# model = model_generator().cuda()  # 将模型移到GPU

# # 设置模型参数的 requires_grad
# if args.method == "finetune":
#     for param in model.parameters():
#         param.requires_grad = True
# else:
#     for param in model.parameters():
#         param.requires_grad = False
# print("trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# # 数据加载
# dataloader = torch.utils.data.DataLoader(Dataset("dataset/lq", "dataset/gt"), batch_size=1, shuffle=False)

# # 学习率设置
# lr = 5e-4 if args.method == "lan" else 5e-6

# # 定义LAN和AFM模型
# class LanWithAFM(torch.nn.Module):
#     def __init__(self, shape, channels, fq_bound=1.0):
#         super(LanWithAFM, self).__init__()
#         self.phi = torch.nn.Parameter(torch.zeros(shape).cuda(), requires_grad=True)
#         self.afm = AFM_B(fq_bound=fq_bound).cuda()  # 将AFM-B移到GPU

#     def forward(self, x):
#         clean = torch.zeros_like(x).cuda()  # 初始化干净图像
#         _, _, fq_mask = self.afm(clean, x)  # 生成掩码
#         adapted_x = x * fq_mask + torch.tanh(self.phi)  # 应用掩码
#         return adapted_x

# # 初始化日志
# logs_key = ["psnr", "ssim"]
# total_logs = {key: [] for key in logs_key}
# inner_loop = 20  # 模拟退火的迭代次数

# # 训练过程
# p_bar = tqdm(dataloader, ncols=100, desc=f"{args.method}_{args.self_loss}")

# for lq, gt in p_bar:
#     lq = lq.cuda()  # 将输入数据移到GPU
#     gt = gt.cuda()  # 将GT数据移到GPU

#     lan = LanWithAFM(lq.shape, lq.shape[1]).cuda() if args.method == "lan" else torch.nn.Identity().cuda()
#     tmp_batch_size = lq.shape[0]
    
#     # 重新初始化模型
#     model = model_generator().cuda()

#     for param in model.parameters():
#         param.requires_grad = args.method == "finetune"

#     # 选择需要优化的参数
#     params = list(lan.parameters()) if args.method == "lan" else list(model.parameters())
#     optimizer = torch.optim.Adam(params, lr=lr)
#     logs = {key: [] for key in logs_key}

#     # 训练迭代
#     for i in range(inner_loop):
#         optimizer.zero_grad()
#         adapted_lq = lan(lq)  # 获取适应后的图像
#         with torch.no_grad():
#             pred = model(adapted_lq).clip(0, 1)  # 使用模型预测并限制输出范围
#         loss = loss_func(adapted_lq, model, i, inner_loop)
#         loss.backward()
#         optimizer.step()
        
#         psnr, ssim = cal_batch_psnr_ssim(pred, gt)
#         for key in logs_key:
#             logs[key].append(locals()[key])

#     else:
#         with torch.no_grad():
#             adapted_lq = lan(lq)
#             pred = model(adapted_lq).clip(0, 1)
#             psnr, ssim = cal_batch_psnr_ssim(pred, gt)
#         for key in logs_key:
#             logs[key].append(locals()[key])

#     # 更新日志
#     for key in logs_key:
#         total_logs[key].extend(np.array(logs[key]).transpose())

#     p_bar.set_postfix(
#         PSNR=f"{np.array(total_logs['psnr']).mean(0)[0]:.2f}->{np.array(total_logs['psnr']).mean(0)[-1]:.2f}",
#         SSIM=f"{np.array(total_logs['ssim']).mean(0)[0]:.3f}->{np.array(total_logs['ssim']).mean(0)[-1]:.3f}"
#     )

# # 保存日志到CSV文件
# df_dict = {
#     "idx": [i for i in range(len(total_logs['psnr'])) for _ in range(inner_loop + 1)],
#     "loop": [i for i in range(inner_loop + 1)] * len(total_logs['psnr']),
# }
# for key in logs_key:
#     df_dict[key] = [value for value_list in total_logs[key] for value in value_list]

# df = pd.DataFrame(df_dict)
# df.to_csv(f"result_{args.method}_{args.self_loss}_origion.csv", index=False)
# print(df.groupby('loop').mean()[['psnr', 'ssim']])



import torch
from data import Dataset
from model import get_model
from metric import cal_batch_psnr_ssim
import pandas as pd
from tqdm import tqdm
import argparse
from adapt import zsn2n, nbr2nbr
import numpy as np
from AFM_B import AFM_B
from torch.cuda.amp import autocast, GradScaler

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, required=True, choices=["finetune", "lan"])
parser.add_argument("--self_loss", type=str, required=True, choices=["nbr2nbr", "zsn2n"])
args = parser.parse_args()

# 选择损失函数
if args.self_loss == "zsn2n":
    loss_func = zsn2n.loss_func
elif args.self_loss == "nbr2nbr":
    loss_func = nbr2nbr.loss_func
else:
    raise NotImplementedError

# 获取模型
model_generator = get_model
model = model_generator().cuda()  # 将模型移到GPU

# 设置模型参数的 requires_grad
if args.method == "finetune":
    for param in model.parameters():
        param.requires_grad = True
else:
    for param in model.parameters():
        param.requires_grad = False
print("trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# 数据加载
dataloader = torch.utils.data.DataLoader(Dataset("dataset/lq", "dataset/gt"), batch_size=1, shuffle=False)

# 学习率设置
lr = 5e-4 if args.method == "lan" else 5e-6

# 定义LAN和AFM模型
class LanWithAFM(torch.nn.Module):
    def __init__(self, shape, channels, fq_bound=1.0):
        super(LanWithAFM, self).__init__()
        self.phi = torch.nn.Parameter(torch.zeros(shape).cuda(), requires_grad=True)
        self.afm = AFM_B(fq_bound=fq_bound).cuda()  # 将AFM-B移到GPU

    def forward(self, x):
        clean = torch.zeros_like(x).cuda()  # 初始化干净图像
        _, _, fq_mask = self.afm(clean, x)  # 生成掩码
        adapted_x = x * fq_mask + torch.tanh(self.phi)  # 应用掩码
        return adapted_x

# 初始化日志
logs_key = ["psnr", "ssim"]
total_logs = {key: [] for key in logs_key}
inner_loop = 20  # 模拟退火的迭代次数

# 自动混合精度
scaler = GradScaler()

# 训练过程
p_bar = tqdm(dataloader, ncols=100, desc=f"{args.method}_{args.self_loss}")

for lq, gt in p_bar:
    lq = lq.cuda()  # 将输入数据移到GPU
    gt = gt.cuda()  # 将GT数据移到GPU

    lan = LanWithAFM(lq.shape, lq.shape[1]).cuda() if args.method == "lan" else torch.nn.Identity().cuda()
    tmp_batch_size = lq.shape[0]
    
    # 重新初始化模型
    model = model_generator().cuda()

    for param in model.parameters():
        param.requires_grad = args.method == "finetune"

    # 选择需要优化的参数
    params = list(lan.parameters()) if args.method == "lan" else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    logs = {key: [] for key in logs_key}

    # 训练迭代
    for i in range(inner_loop):
        optimizer.zero_grad()

        with autocast():  # 启用自动混合精度
            adapted_lq = lan(lq)  # 获取适应后的图像
            with torch.no_grad():
                pred = model(adapted_lq).clip(0, 1)  # 使用模型预测并限制输出范围

            loss = loss_func(adapted_lq, model, i, inner_loop)

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 计算PSNR和SSIM
        psnr, ssim = cal_batch_psnr_ssim(pred, gt)
        logs["psnr"].append(psnr)
        logs["ssim"].append(ssim)

    else:
        with torch.no_grad():
            adapted_lq = lan(lq)
            pred = model(adapted_lq).clip(0, 1)
            psnr, ssim = cal_batch_psnr_ssim(pred, gt)
        logs["psnr"].append(psnr)
        logs["ssim"].append(ssim)

    # 更新日志
    for key in logs_key:
        total_logs[key].extend(np.array(logs[key]).transpose())

    p_bar.set_postfix(
        PSNR=f"{np.array(total_logs['psnr']).mean(0)[0]:.2f}->{np.array(total_logs['psnr']).mean(0)[-1]:.2f}",
        SSIM=f"{np.array(total_logs['ssim']).mean(0)[0]:.3f}->{np.array(total_logs['ssim']).mean(0)[-1]:.3f}"
    )

    # 清理缓存以避免内存泄漏
    torch.cuda.empty_cache()

# 保存日志到CSV文件
df_dict = {
    "idx": [i for i in range(len(total_logs['psnr'])) for _ in range(inner_loop + 1)],
    "loop": [i for i in range(inner_loop + 1)] * len(total_logs['psnr']),
}
for key in logs_key:
    df_dict[key] = [value for value_list in total_logs[key] for value in value_list]

df = pd.DataFrame(df_dict)
df.to_csv(f"result_{args.method}_{args.self_loss}_origion.csv", index=False)

# 打印每个循环的平均PSNR和SSIM
print(df.groupby('loop').mean()[['psnr', 'ssim']])
