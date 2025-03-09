import torch
from data import Dataset
from model import get_model
from metric import cal_batch_psnr_ssim
import pandas as pd
from tqdm import tqdm
import argparse
from adapt import zsn2n, nbr2nbr
import numpy as np

# 模拟退火相关函数
def metropolis_criterion(delta_E, T):
    """Metropolis准则：计算接受概率"""
    if delta_E < 0:
        return 1.0
    else:
        return np.exp(-delta_E / T)

def update_temperature(T, alpha=0.99):
    """温度更新函数：指数衰减"""
    return T * alpha

# 修改 LAN 类以支持模拟退火
class Lan(torch.nn.Module):
    def __init__(self, shape, initial_temp=1.0, alpha=0.99):
        super(Lan, self).__init__()
        self.phi = torch.nn.Parameter(torch.zeros(shape).cuda(), requires_grad=True)  # 参数放在GPU上
        self.T = initial_temp  # 初始温度
        self.alpha = alpha  # 降温系数

    def forward(self, x):
        return x + torch.tanh(self.phi)

# 主程序
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, required=True, choices=["finetune", "lan"])
parser.add_argument("--self_loss", type=str, required=True, choices=["nbr2nbr", "zsn2n"])
args = parser.parse_args()

if args.self_loss == "zsn2n":
    loss_func = zsn2n.loss_func
elif args.self_loss == "nbr2nbr":
    loss_func = nbr2nbr.loss_func
else:
    raise NotImplementedError

model_generator = get_model
model = model_generator().cuda()  # 确保模型在GPU上

# 设置模型参数的 requires_grad
if args.method == "finetune":
    for param in model.parameters():
        param.requires_grad = True
else:
    for param in model.parameters():
        param.requires_grad = False
print("trainable model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

dataloader = torch.utils.data.DataLoader(Dataset("dataset/lq", "dataset/gt"), batch_size=1, shuffle=False)
lr = 5e-4 if args.method == "lan" else 5e-6

logs_key = ["psnr", "ssim"]
total_logs = {key: [] for key in logs_key}
inner_loop = 20  # 模拟退火的迭代次数
p_bar = tqdm(dataloader, ncols=100, desc=f"{args.method}_{args.self_loss}")

for lq, gt in p_bar:
    lq = lq.cuda()  # 将输入数据转移到GPU
    gt = gt.cuda()  # 将GT数据转移到GPU
    lan = Lan(lq.shape, initial_temp=1.0, alpha=0.99).cuda() if args.method == "lan" else torch.nn.Identity().cuda()  # lan也放到GPU上
    tmp_batch_size = lq.shape[0]
    model = model_generator().cuda()  # 确保模型是GPU版本

    for param in model.parameters():
        param.requires_grad = args.method == "finetune"

    params = list(lan.parameters()) if args.method == "lan" else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    logs = {key: [] for key in logs_key}

    prev_loss = float("inf")
    prev_phi = None
    for i in range(inner_loop):
        optimizer.zero_grad()
        adapted_lq = lan(lq)  
        with torch.no_grad():
            pred = model(adapted_lq).clip(0, 1)
        loss = loss_func(adapted_lq, model, i, inner_loop)
        loss.backward()
        optimizer.step()

        # 模拟退火逻辑
        if args.method == "lan":
            current_loss = loss.item()
            if i > 0:
                delta_E = current_loss - prev_loss
                accept_prob = metropolis_criterion(delta_E, lan.T)
                if np.random.rand() > accept_prob:
                    # 恢复到之前的参数
                    lan.phi.data = prev_phi.clone()
                    loss = prev_loss
                else:
                    prev_phi = lan.phi.data.clone()
            prev_loss = current_loss
            lan.T = update_temperature(lan.T, lan.alpha)

        psnr, ssim = cal_batch_psnr_ssim(pred, gt)
        for key in logs_key:
            logs[key].append(locals()[key])

        # 打印每次 inner_loop 的 PSNR 和 SSIM 值
        print(f"Loop {i+1}/{inner_loop}, PSNR: {psnr[-1]:.2f}, SSIM: {ssim[-1]:.4f}")

    for key in logs_key:
        total_logs[key].extend(np.array(logs[key]).transpose())
    p_bar.set_postfix(
        PSNR=f"{np.array(total_logs['psnr']).mean(0)[0]:.2f}->{np.array(total_logs['psnr']).mean(0)[-1]:.2f}",
        SSIM=f"{np.array(total_logs['ssim']).mean(0)[0]:.3f}->{np.array(total_logs['ssim']).mean(0)[-1]:.3f}",
        Temp=f"{lan.T:.4f}" if args.method == "lan" else ""
    )

# 在循环外部汇总日志数据并保存到文件
df_dict = {
    "idx": [i for i in range(len(total_logs['psnr'])) for _ in range(inner_loop+1)],
    "loop": [i for i in range(inner_loop+1)] * len(total_logs['psnr']),
}
for key in logs_key:
    df_dict[key] = [value for value_list in total_logs[key] for value in value_list]
df_dict["temp"] = [lan.T] * (inner_loop + 1) if args.method == "lan" else [0] * (inner_loop + 1)
df = pd.DataFrame(df_dict)
df.to_csv(f"result_{args.method}_{args.self_loss}_sa.csv", index=False)
print(df.groupby('loop').mean()[['psnr', 'ssim']])

#python "D:\国防科技大学\大三\AIBD\肺分割论文\image denoise\代码\LAN-master\LAN-master\mainfire.py" --method lan --self_loss zsn2n
