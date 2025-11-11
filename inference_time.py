import torch
import torch.nn as nn
import time
import os
from thop import profile, clever_format
from thop import profile, clever_format
from models.RADAR import RADAR
import torch

# ====== 示例模型（U-Net）======
# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1):
#         super(UNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2)
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, out_channels, 1)
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

if __name__ == '__main__':
# ====== 初始化 ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RADAR().to(device)
    input_tensor = torch.randn(1, 3, 1120, 1120).to(device)

    # ====== 1. 计算 FLOPs & Params ======
    flops_raw, params_raw = profile(model, inputs=(input_tensor,), verbose=False)
    flops_str, params_str = clever_format([flops_raw, params_raw], "%.3f")

    # ====== 2. 理论模型大小 ======
    theoretical_size_MB = params_raw * 4 / 1024 / 1024  # float32

    # ====== 3. 推理时间 & GPU 内存占用 ======
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            _ = model(input_tensor)
            torch.cuda.synchronize()
        end_time = time.time()

    avg_time_ms = (end_time - start_time) / 100 * 1000

    # 获取显存使用
    max_memory_MB = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # ====== 4. 实际权重文件大小 ======
    weight_path = "temp_weights.pth"
    torch.save(model.state_dict(), weight_path)
    actual_size_MB = os.path.getsize(weight_path) / 1024 / 1024
    os.remove(weight_path)

    # ====== 5. 输出结果 ======
    print("===== Model Statistics =====")
    print(f"FLOPs: {flops_str}")
    print(f"Params: {params_str}")
    print(f"Theoretical Model Size: {theoretical_size_MB:.2f} MB (float32)")
    print(f"Actual Weight File Size: {actual_size_MB:.2f} MB")
    print(f"Average Inference Time: {avg_time_ms:.3f} ms")
    print(f"Max GPU Memory Usage: {max_memory_MB:.2f} MB")
