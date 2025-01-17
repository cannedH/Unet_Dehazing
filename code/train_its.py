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



# import torchvision.transforms as transforms


class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.clear_images = sorted(os.listdir(clear_dir))  # 清晰图片
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform

        # 缓存 hazy_dir 中的文件到字典中
        self.hazy_files_dict = {}
        for file in os.listdir(self.hazy_dir):
            base = file.split('_')[0]  # 提取文件名前缀，例如 "1"。
            if base not in self.hazy_files_dict:
                self.hazy_files_dict[base] = []
            self.hazy_files_dict[base].append(file)

    def __len__(self):
        return len(self.clear_images) * 10

    def __getitem__(self, idx):
        # 计算清晰图像和对应的有雾图像编号
        clear_idx = idx // 10  # 确定清晰图像的索引
        hazy_idx = idx % 10  # 确定有雾图像编号，从0到9

        # 加载清晰图像
        clear_image_name = self.clear_images[clear_idx]
        clear_img_path = os.path.join(self.clear_dir, clear_image_name)
        clear_img = Image.open(clear_img_path).convert("RGB")

        # 提取基础编号（去掉扩展名）
        base_name = os.path.splitext(clear_image_name)[0]

        # 查找对应的 hazy 文件
        if base_name not in self.hazy_files_dict:
            raise FileNotFoundError(f"No hazy files found for {base_name}.png")

        hazy_files = self.hazy_files_dict[base_name]
        if hazy_idx >= len(hazy_files):
            raise IndexError(f"Hazy index {hazy_idx} out of range for {base_name}")

        hazy_img_path = os.path.join(self.hazy_dir, hazy_files[hazy_idx])
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
    # transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 加载数据集
hazy_dir = r"F:\datasets\hazy_its"
clear_dir = r"F:\datasets\clear_its"
dataset = DehazeDataset(hazy_dir, clear_dir, transform=transform)

# 数据集划分为训练集和验证集
validation_size = 990
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
checkpoint_path = r"G:\PythonProject\Unet_dehazing\model_\indoor\dehaze_model_epoch_last.pth"

logging.basicConfig(filename="training_its.log", level=logging.INFO)
warnings.simplefilter("ignore", FutureWarning)

start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch + 1}")
else:
    print("No checkpoint found, starting training from scratch.")

# 定义 PSNR 和 SSIM 指标
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# 训练循环
def train():
    num_epochs = 100
    #num_batches = len(train_dataloader)
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

                # print("www",outputs.min(),outputs.max(),clear.min(),clear.max())

                # assert outputs.min() >= 0 and outputs.max() <= 1, "Output not normalized to [0, 1]"
                # assert clear.min() >= 0 and clear.max() <= 1, "Clear image not normalized to [0, 1]"
                # 更新指标
                psnr_metric.update(outputs, clear)
                ssim_metric.update(outputs, clear)

                # 累加损失
                running_loss += loss.item()

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # visualize_output(hazy, clear, outputs, epoch + 1)
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
            save_dir = r"G:\PythonProject\Unet_dehazing\model_\indoor"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"dehaze_model_epoch_{epoch + 1}.pth"))
            print(f"Model saved at epoch {epoch + 1}")

        save_dir = r"G:\PythonProject\Unet_dehazing\model_\indoor"
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
