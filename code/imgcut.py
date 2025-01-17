import os
from PIL import Image


def crop_images(input_dir, gt_output_dir, noise_output_dir, crop_size=512):
    """
    裁剪输入目录中的 GT 和 NOISY 图像，保存为裁剪后的子图。

    Args:
        input_dir (str): 输入数据集主目录路径。
        gt_output_dir (str): 输出 GT 裁剪图像的保存路径。
        noise_output_dir (str): 输出 NOISE 裁剪图像的保存路径。
        crop_size (int): 裁剪的图像大小，默认 512。
    """
    # 创建 GT 和 NOISE 输出目录
    os.makedirs(gt_output_dir, exist_ok=True)
    os.makedirs(noise_output_dir, exist_ok=True)

    # 遍历输入目录中的子文件夹
    crop_count = 1  # 全局计数器，用于确保文件名唯一
    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        gt_path = None
        noisy_path = None

        # 寻找 GT 和 NOISY 图像
        for file in os.listdir(subfolder_path):
            if file.startswith("GT_") and file.endswith(".PNG"):
                gt_path = os.path.join(subfolder_path, file)
            elif file.startswith("NOISY_") and file.endswith(".PNG"):
                noisy_path = os.path.join(subfolder_path, file)

        # 跳过未找到成对图像的情况
        if not gt_path or not noisy_path:
            print(f"Skipped folder {subfolder} due to missing GT or NOISY image.")
            continue

        # 打开 GT 和 NOISY 图像
        gt_image = Image.open(gt_path)
        noisy_image = Image.open(noisy_path)

        # 确保 GT 和 NOISY 图像大小一致
        if gt_image.size != noisy_image.size:
            print(f"Skipped folder {subfolder} due to size mismatch between GT and NOISY images.")
            continue

        width, height = gt_image.size

        # 遍历裁剪区域
        for top in range(0, height - crop_size + 1, crop_size):
            for left in range(0, width - crop_size + 1, crop_size):
                # 定义裁剪区域
                box = (left, top, left + crop_size, top + crop_size)

                # 裁剪图像
                gt_crop = gt_image.crop(box)
                noisy_crop = noisy_image.crop(box)

                # 保存裁剪后的图像
                gt_crop.save(os.path.join(gt_output_dir, f"GT_{crop_count}.PNG"))
                noisy_crop.save(os.path.join(noise_output_dir, f"NOISE_{crop_count}.PNG"))
                crop_count += 1

        print(f"Processed folder {subfolder}")

if __name__ == "__main__":
    input_dir = r"/Data"  # 输入主文件夹路径
    gt_output_dir = r"F:\datasets\SIDD\GT"  # 输出 GT 裁剪图像的保存路径
    noise_output_dir = r"F:\datasets\SIDD\NOISE"  # 输出 NOISE 裁剪图像的保存路径
    crop_size = 512  # 裁剪块大小

    crop_images(input_dir, gt_output_dir, noise_output_dir, crop_size)


#
# import os
# from PIL import Image
#
#
# def crop_images_to_multiple_of_8(input_dir, output_dir=None):
#     """
#     将指定文件夹中的所有图片裁剪为长宽均为8的倍数。
#
#     :param input_dir: 输入图片文件夹路径
#     :param output_dir: 输出图片文件夹路径。如果为 None，覆盖输入文件夹的图片。
#     """
#     # 如果未指定输出文件夹，则使用输入文件夹作为输出文件夹
#     if output_dir is None:
#         output_dir = input_dir
#     else:
#         os.makedirs(output_dir, exist_ok=True)
#
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(input_dir):
#         file_path = os.path.join(input_dir, filename)
#
#         # 检查文件是否为图片
#         try:
#             with Image.open(file_path) as img:
#                 # 获取原始尺寸
#                 width, height = img.size
#
#                 # 计算裁剪后的尺寸
#                 new_width = (width // 8) * 8
#                 new_height = (height // 8) * 8
#
#                 # 裁剪图片
#                 cropped_img = img.crop((0, 0, new_width, new_height))
#
#                 # 保存裁剪后的图片
#                 output_path = os.path.join(output_dir, filename)
#                 cropped_img.save(output_path, quality=100)
#
#                 print(f"Cropped and saved: {output_path}")
#         except Exception as e:
#             print(f"Error processing file {file_path}: {e}")
#
#
#
# input_directory = r"F:\datasets\hazy_ots_"
# output_directory = r"F:\datasets\hazy_ots"
#
# crop_images_to_multiple_of_8(input_directory, output_directory)
