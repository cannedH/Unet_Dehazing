from net import DeHazing
import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 创建模型实例并加载权重
model = DeHazing()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(
    torch.load(r"model\sidd_denoise.pth", map_location=device))
model.to(device)
model.eval()  # 设置模型为评估模式

# 数据集路径
clear_images_path = r"data\SIDD\GT_test"  # 清晰图像目录
noisy_images_path = r"data\SIDD\NOISE_test"  # 噪声图像目录

# 统计 PSNR 和 SSIM
total_psnr, total_ssim, num_samples = 0, 0, 0

# 获取清晰图像列表
clear_image_files = [f for f in os.listdir(clear_images_path) if f.endswith('.PNG')]

# 遍历清晰图像文件夹，并添加进度条
for clear_image_name in tqdm(clear_image_files, desc="Processing images"):
    clear_image_path = os.path.join(clear_images_path, clear_image_name)

    # 读取清晰图像
    clear_image = cv2.imread(clear_image_path)
    clear_image = cv2.cvtColor(clear_image, cv2.COLOR_BGR2RGB)

    # 遍历与该清晰图像对应的噪声图像
    noisy_image_name = clear_image_name.replace("GT", "NOISE")  # 假设噪声图像的命名规则与清晰图像一致
    noisy_image_path = os.path.join(noisy_images_path, noisy_image_name)

    if not os.path.exists(noisy_image_path):
        print(f"File {noisy_image_path} does not exist.")
        continue

    # 读取噪声图像
    noisy_image = cv2.imread(noisy_image_path)
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

    # 预处理噪声图像
    input_image = noisy_image / 255.0  # 归一化到 [0, 1]
    input_image = np.transpose(input_image, (2, 0, 1))  # 转换为 CxHxW
    input_image = torch.FloatTensor(input_image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        dehazed_image = model(input_image)
    dehazed_image = torch.clamp(dehazed_image, 0, 1)

    # 后处理去雾结果
    dehazed_image = dehazed_image.squeeze(0).cpu().numpy()
    dehazed_image = np.transpose(dehazed_image, (1, 2, 0))
    dehazed_image = (dehazed_image * 255).astype(np.uint8)

    # 计算 PSNR 和 SSIM
    current_psnr = psnr(clear_image, dehazed_image, data_range=255)
    current_ssim = ssim(clear_image, dehazed_image, data_range=255, win_size=3, channel_axis=2)
    total_psnr += current_psnr
    total_ssim += current_ssim
    num_samples += 1

# 计算平均 PSNR 和 SSIM
average_psnr = total_psnr / num_samples
average_ssim = total_ssim / num_samples

print(f"\nAverage PSNR: {average_psnr:.2f}")
print(f"Average SSIM: {average_ssim:.4f}")
