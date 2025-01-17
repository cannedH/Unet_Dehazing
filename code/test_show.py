from net import DeHazing
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建模型实例
model = DeHazing()

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def crop_to_multiple_of_8(image):
    height, width = image.shape[:2]
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    return image[:new_height, :new_width]

# 加载模型权重到指定设备
# 读取有雾图片
# model.load_state_dict(torch.load(r"model/its_indoor.pth", map_location=device))
# image_path = r"G:\PythonProject\Unet_dehazing\SOTS\indoor\hazy\1420_10.png"
# model.load_state_dict(torch.load(r"\model\ots_outdoor", map_location=device))
# image_path = r"G:\PythonProject\Unet_dehazing\SOTS\outdoor\hazy\0018_0.9_0.2.jpg"
model.load_state_dict(torch.load(r"model\sidd_denoise.pth", map_location=device))
image_path = r""
output_image_path = r""
model.to(device)  # 将模型移动到 GPU

hazy_image = cv2.imread(image_path)
hazy_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式

hazy_image = crop_to_multiple_of_8(hazy_image)

# 预处理图片
input_image = hazy_image / 255.0  # 归一化到 [0, 1]
input_image = np.transpose(input_image, (2, 0, 1))  # 转换为 CxHxW
input_image = torch.FloatTensor(input_image).unsqueeze(0).to(device)

# 推理
with torch.no_grad():
    dehazed_image = model(input_image)

# 将输出图像的值限制在 [0, 1] 范围内
dehazed_image = torch.clamp(dehazed_image, 0, 1)

# 后处理图片
dehazed_image = dehazed_image.squeeze(0).cpu().numpy()
dehazed_image = np.transpose(dehazed_image, (1, 2, 0))  # 转回 HxWxC
dehazed_image = (dehazed_image * 255).astype(np.uint8)  # 恢复到 0-255 之间

cv2.imwrite(output_image_path, cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR))  # 转为BGR格式保存

# 显示原图和去雾后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Hazy Image')
plt.imshow(hazy_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Dehazed Image')
plt.imshow(dehazed_image)
plt.axis('off')

plt.show()