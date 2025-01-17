from torch.utils.data import DataLoader, Dataset, random_split
from net import DeHazing
import os
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import logging
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import warnings


# 自定义数据集类
class DenoiseDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, transform=None):
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.data = []

        # 遍历 GT 和 NOISE 文件夹
        gt_files = sorted(os.listdir(gt_dir))
        noisy_files = sorted(os.listdir(noisy_dir))

        # 确保 GT 和 NOISE 图像数量相同
        if len(gt_files) != len(noisy_files):
            raise ValueError(f"GT and NOISE directories have different number of files: "
                             f"{len(gt_files)} vs {len(noisy_files)}")

        # 将文件对保存到数据集列表中
        for gt_file, noisy_file in zip(gt_files, noisy_files):
            if gt_file.endswith('.PNG') and noisy_file.endswith('.PNG'):
                self.data.append((os.path.join(gt_dir, gt_file), os.path.join(noisy_dir, noisy_file)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gt_path, noisy_path = self.data[idx]

        # 读取图像
        gt_img = Image.open(gt_path).convert("RGB")
        noisy_img = Image.open(noisy_path).convert("RGB")

        # 应用数据增强
        if self.transform:
            gt_img = self.transform(gt_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, gt_img


# 模型评估函数
def evaluate_model(model, dataloader, device, psnr_metric, ssim_metric):
    model.eval()
    psnr_metric.reset()
    ssim_metric.reset()

    with torch.no_grad():
        for noisy, gt in dataloader:
            noisy = noisy.to(device)
            gt = gt.to(device)
            output = model(noisy)

            psnr_metric.update(output, gt)
            ssim_metric.update(output, gt)

    avg_psnr = psnr_metric.compute().item()
    avg_ssim = ssim_metric.compute().item()
    return avg_psnr, avg_ssim


# 定义数据转换
transform = transforms.Compose([transforms.ToTensor()])

# 定义文件夹路径
gt_dir = r"F:\datasets\SIDD\GT"  # GT 图像文件夹路径
noisy_dir = r"F:\datasets\SIDD\NOISE"  # NOISE 图像文件夹路径

# 加载数据集
dataset = DenoiseDataset(gt_dir, noisy_dir, transform=transform)

# 数据集划分为训练集和验证集
validation_size = 1000
train_size = len(dataset) - validation_size
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
checkpoint_path = r"G:\PythonProject\Unet_dehazing\model_\denoising\denoise_model_epoch_last.pth"

logging.basicConfig(filename="training_denoising.log", level=logging.INFO)
warnings.simplefilter("ignore", FutureWarning)

start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']  # 恢复训练时从哪里开始
    print(f"Resuming training from epoch {start_epoch + 1}")
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
            for noisy, gt in train_dataloader:
                noisy = noisy.to(device)
                gt = gt.to(device)

                # 前向传播
                outputs = model(noisy)
                loss = criterion(outputs, gt)

                # 计算 PSNR 和 SSIM
                psnr_metric.update(outputs, gt)
                ssim_metric.update(outputs, gt)

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
            save_dir = r"G:\PythonProject\Unet_dehazing\model_\denoising"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"denoise_model_epoch_{epoch + 1}.pth"))
            print(f"Model saved at epoch {epoch + 1}")

        save_dir = r"G:\PythonProject\Unet_dehazing\model_\denoising"
        checkpoint_path = os.path.join(save_dir, "denoise_model_epoch_last.pth")
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
