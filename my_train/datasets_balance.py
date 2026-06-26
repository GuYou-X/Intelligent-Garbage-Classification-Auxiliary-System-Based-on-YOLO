import os
import random
import shutil
from collections import defaultdict

# ===================== 路径 =====================
DATASET_ROOT = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\cleaned_garbage_datasets"
train_img_dir = os.path.join(DATASET_ROOT, "images", "train")
train_lab_dir = os.path.join(DATASET_ROOT, "labels", "train")
val_img_dir = os.path.join(DATASET_ROOT, "images", "val")
val_lab_dir = os.path.join(DATASET_ROOT, "labels", "val")

NEW_DATA_ROOT = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\balanced_garbage_datasets"
# =================================================

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
TRAIN_RATIO = 0.8  # 每个类别内部8:2划分


# 读取图片-标签对
def load_image_label_pairs(img_dir, lab_dir):
    pairs = []
    for img_name in os.listdir(img_dir):
        ext = os.path.splitext(img_name)[-1].lower()
        if ext not in IMAGE_EXTS:
            continue
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        lab_path = os.path.join(lab_dir, base + ".txt")
        if os.path.exists(lab_path):
            pairs.append((img_path, lab_path))
    return pairs


# 加载全部数据
all_train = load_image_label_pairs(train_img_dir, train_lab_dir)
all_val = load_image_label_pairs(val_img_dir, val_lab_dir)
all_data = all_train + all_val
print(f"✅ 原始总图片数：{len(all_data)}")

# 创建输出目录
for d in [
    os.path.join(NEW_DATA_ROOT, "images", "train"),
    os.path.join(NEW_DATA_ROOT, "labels", "train"),
    os.path.join(NEW_DATA_ROOT, "images", "val"),
    os.path.join(NEW_DATA_ROOT, "labels", "val"),
]:
    os.makedirs(d, exist_ok=True)

# ===================== 原生Python分层8:2划分 =====================
# 1. 按类别分组（保证每个类别都有数据）
class_to_files = defaultdict(list)
for img_p, lab_p in all_data:
    try:
        with open(lab_p, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        # 提取第一个类别作为主类别（多标签图取第一个，不影响分层）
        main_cls = lines[0].split()[0] if lines else "-1"
        class_to_files[main_cls].append((img_p, lab_p))
    except:
        class_to_files["-1"].append((img_p, lab_p))

# 2. 每个类别内部8:2划分，收集最终train/val列表
train_data = []
val_data = []

for cls, file_list in class_to_files.items():
    # 打乱该类别下的所有图片
    random.shuffle(file_list)
    # 计算8:2划分点
    split_idx = int(len(file_list) * TRAIN_RATIO)
    # 划分训练/验证
    train_data.extend(file_list[:split_idx])
    val_data.extend(file_list[split_idx:])

# 3. 再次全局打乱（避免类别顺序影响训练）
random.shuffle(train_data)
random.shuffle(val_data)

# 4. 复制文件到新目录
def copy_files(data_list, split):
    img_out_dir = os.path.join(NEW_DATA_ROOT, "images", split)
    lab_out_dir = os.path.join(NEW_DATA_ROOT, "labels", split)
    count = 0
    for img_p, lab_p in data_list:
        shutil.copy(img_p, os.path.join(img_out_dir, os.path.basename(img_p)))
        shutil.copy(lab_p, os.path.join(lab_out_dir, os.path.basename(lab_p)))
        count += 1
    return count

# 执行复制
train_cnt = copy_files(train_data, "train")
val_cnt = copy_files(val_data, "val")

# ===================== 统计输出 =====================
print("-" * 60)
print(f"✅ 分层8:2划分完成！")
print(f"训练集图片：{train_cnt} 张")
print(f"验证集图片：{val_cnt} 张")
print(f"总图片：{train_cnt + val_cnt} 张（与原始数据完全一致）")
print(f"输出路径：{NEW_DATA_ROOT}")

# 额外统计每个类别的数量（方便你检查）
print("\n📊 各类别数量统计：")
for cls, file_list in class_to_files.items():
    split_idx = int(len(file_list) * TRAIN_RATIO)
    print(f"类别 {cls:>2} | 训练 {split_idx:>4} | 验证 {len(file_list)-split_idx:>3} | 总数 {len(file_list):>4}")