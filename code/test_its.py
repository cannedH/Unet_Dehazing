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
    torch.load(r"model/its_indoor.pth", map_location=device))
model.to(device)
model.eval()  # 设置模型为评估模式

# 数据集路径
hazy_images_path = r"data\SOTS\indoor\hazy"
clear_images_path = r"data\SOTS\indoor\gt"

# 统计 PSNR 和 SSIM
total_psnr, total_ssim, num_samples = 0, 0, 0

# 获取清晰图像列表
clear_image_files = [f for f in os.listdir(clear_images_path) if f.endswith('.png')]

# 计算裁剪的大小（8的倍数）
def crop_to_multiple_of_8(image):
    height, width, _ = image.shape
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    cropped_image = image[:new_height, :new_width]
    return cropped_image

# 遍历清晰图像文件夹，并添加进度条
for clear_image_name in tqdm(clear_image_files, desc="Processing images"):
    clear_image_path = os.path.join(clear_images_path, clear_image_name)

    # 读取清晰图像
    clear_image = cv2.imread(clear_image_path)
    clear_image = cv2.cvtColor(clear_image, cv2.COLOR_BGR2RGB)

    # 裁剪清晰图像为 8 的倍数
    clear_image = crop_to_multiple_of_8(clear_image)

    # 遍历与该清晰图像对应的10张有雾图片
    for i in range(1, 11):
        hazy_image_name = f"{clear_image_name.split('.')[0]}_{i}.png"
        hazy_image_path = os.path.join(hazy_images_path, hazy_image_name)

        if not os.path.exists(hazy_image_path):
            print(f"File {hazy_image_path} does not exist.")
            continue

        # 读取有雾图像
        hazy_image = cv2.imread(hazy_image_path)
        hazy_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB)

        # 对有雾图像裁剪为与清晰图像相同的区域
        hazy_image = crop_to_multiple_of_8(hazy_image)

        # 预处理有雾图像
        input_image = hazy_image / 255.0  # 归一化到 [0, 1]
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

        # 计算 PSNR 和 SSIM，指定 win_size 和 channel_axis
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
