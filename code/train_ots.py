from torch.utils.data import DataLoader, Dataset, random_split
from net import DeHazing
import os
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import logging
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import warnings


# 自定义数据集类
class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.clear_images = sorted(os.listdir(clear_dir))  # 清晰图片
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform

    def __len__(self):
        return len(self.clear_images) * 35  # 每张清晰图对应 35 张有雾图

    def __getitem__(self, idx):
        clear_idx = idx // 35  # 确定清晰图像的索引
        param_idx = idx % 35  # 计算出具体的编号索引

        # 计算第二项和第三项的编号
        second_idx = param_idx // 7  # 第二项（0.8 到 1）
        third_idx = param_idx % 7  # 第三项（0.04 到 0.2）

        # 加载清晰图像
        clear_image_name = self.clear_images[clear_idx]
        clear_img_path = os.path.join(self.clear_dir, clear_image_name)
        clear_img = Image.open(clear_img_path).convert("RGB")

        # 提取基础编号
        base_name = os.path.splitext(clear_image_name)[0]

        # 构建有雾图像文件名
        second_values = ["0.8", "0.85", "0.9", "0.95", "1"]
        third_values = ["0.04", "0.06", "0.08", "0.1", "0.12", "0.16", "0.2"]

        second_value = second_values[second_idx]
        third_value = third_values[third_idx]

        hazy_image_name = f"{base_name}_{second_value}_{third_value}.jpg"
        hazy_img_path = os.path.join(self.hazy_dir, hazy_image_name)

        if not os.path.exists(hazy_img_path):
            raise FileNotFoundError(f"Hazy image {hazy_image_name} not found")

        hazy_img = Image.open(hazy_img_path).convert("RGB")

        # 应用数据增强
        if self.transform:
            clear_img = self.transform(clear_img)
            hazy_img = self.transform(hazy_img)

        return hazy_img, clear_img


# 模型评估函数
def evaluate_model(model, dataloader, device, psnr_metric, ssim_metric):
    model.eval()
    psnr_metric.reset()
    ssim_metric.reset()

    with torch.no_grad():
        for hazy, clear in dataloader:
            hazy = hazy.to(device)
            clear = clear.to(device)
            output = model(hazy)

            psnr_metric.update(output, clear)
            ssim_metric.update(output, clear)

        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        return avg_psnr, avg_ssim


# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载数据集
hazy_dir = r"F:\datasets\hazy_ots"
clear_dir = r"F:\datasets\clear_ots"
dataset = DehazeDataset(hazy_dir, clear_dir, transform=transform)

# 数据集划分为训练集和验证集
validation_size = int(len(dataset)*0.1)
train_size = len(dataset)-validation_size
train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

# 模型、损失函数和优化器
model = DeHazing()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = torch.nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
checkpoint_path = r"G:\PythonProject\Unet_dehazing\model_\outdoor\dehaze_model_epoch_last.pth"

logging.basicConfig(filename="training_ots.log", level=logging.INFO)
warnings.simplefilter("ignore", FutureWarning)

start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # 恢复训练时从哪里开始
    # print(f"Resuming training from epoch {start_epoch + 1}")
else:
    print("No checkpoint found, starting training from scratch.")


psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


# 训练循环
def train():
    num_epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        psnr_metric.reset()
        ssim_metric.reset()

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for hazy, clear in train_dataloader:
                hazy = hazy.to(device)
                clear = clear.to(device)

                # 前向传播
                outputs = model(hazy)
                loss = criterion(outputs, clear)

                # print("www", outputs.min(), outputs.max(), clear.min(), clear.max())

                # 计算 PSNR 和 SSIM
                psnr_metric.update(outputs, clear)
                ssim_metric.update(outputs, clear)

                # 累加损失和指标
                running_loss += loss.item()

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新进度条
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = running_loss / (pbar.n + 1)
                avg_psnr = psnr_metric.compute().item()
                avg_ssim = ssim_metric.compute().item()
                pbar.set_postfix(lr=current_lr, loss=avg_loss, psnr=avg_psnr, ssim=avg_ssim)
                pbar.update(1)

        # 每个 epoch 结束后在验证集上评估模型
        avg_psnr, avg_ssim = evaluate_model(model, val_dataloader, device, psnr_metric, ssim_metric)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation PSNR: {avg_psnr:.4f}, Validation SSIM: {avg_ssim:.4f}")

        scheduler.step()

        if (epoch + 1) % 3 == 0:
            save_dir = r"G:\PythonProject\Unet_dehazing\model_\outdoor"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"dehaze_model_epoch_{epoch + 1}.pth"))
            print(f"Model saved at epoch {epoch + 1}")

        save_dir = r"G:\PythonProject\Unet_dehazing\model_\outdoor"
        checkpoint_path = os.path.join(save_dir, "dehaze_model_epoch_last.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(train_dataloader),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Last model saved at epoch {epoch + 1}")
        logging.info(f"Epoch {epoch + 1}: Loss = {avg_loss}, PSNR = {avg_psnr}, SSIM = {avg_ssim}")

        torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
