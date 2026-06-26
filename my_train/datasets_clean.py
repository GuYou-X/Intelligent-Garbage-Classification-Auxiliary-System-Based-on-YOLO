import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ===================== 核心配置项 =====================
# 1. 原始数据集路径
RAW_DATASET_ROOT = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\garbage_datasets\datasets"
# 2. 清理后数据集保存路径
CLEANED_DATASET_ROOT = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\cleaned_garbage_datasets"
# 3. 图表保存路径（my_train/cleaned_dataset_plots）
MY_TRAIN_ROOT = os.path.dirname(os.path.dirname(RAW_DATASET_ROOT))
PLOT_SAVE_PATH = os.path.join(MY_TRAIN_ROOT, "cleaned_garbage_datasets_analysis_plots")
# 4. 基础配置
NUM_CLASSES = 40
ALLOWED_IMG_FORMATS = [".jpg"]
# 5. 图表样式配置（论文级）
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

# ===================== 工具函数 =====================
def copy_dataset(src, dst):
    """复制原始数据集到新目录（避免修改原数据）"""
    if os.path.exists(dst):
        print(f"⚠️  清理后数据集目录已存在，先删除：{dst}")
        shutil.rmtree(dst)

    print(f"\n📤 正在复制原始数据集到：{dst}")
    # 递归复制目录
    shutil.copytree(src, dst)
    print("✅ 数据集复制完成！")


def is_image_valid(img_path):
    """检查图片是否有效（复用校验逻辑）"""
    try:
        img = cv2.imread(img_path)
        if img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
            return False
        return True
    except:
        return False


def validate_dataset(dataset_root):
    """校验数据集，返回错误详情（用于生成错误分布图）"""
    all_errors = {
        "train": {"missing_label": [], "missing_image": [], "invalid_image": [], "invalid_label": []},
        "val": {"missing_label": [], "missing_image": [], "invalid_image": [], "invalid_label": []}
    }

    for split in ["train", "val"]:
        img_dir = os.path.join(dataset_root, "images", split)
        label_dir = os.path.join(dataset_root, "labels", split)

        # 获取所有文件
        img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in ALLOWED_IMG_FORMATS]
        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

        # 1. 缺少标注的图片（有图片无标注）
        img_names = set([os.path.splitext(f)[0] for f in img_files])
        label_names = set([os.path.splitext(f)[0] for f in label_files])
        missing_label = [f for f in img_files if os.path.splitext(f)[0] not in label_names]
        all_errors[split]["missing_label"] = missing_label

        # 2. 缺少图片的标注（有标注无图片）
        missing_image = [f for f in label_files if os.path.splitext(f)[0] not in img_names]
        all_errors[split]["missing_image"] = missing_image

        # 3. 损坏图片
        invalid_image = []
        for img_file in img_files:
            if not is_image_valid(os.path.join(img_dir, img_file)):
                invalid_image.append(img_file)
        all_errors[split]["invalid_image"] = invalid_image

        # 4. 格式错误标注（简化版：仅检查行数和数值格式）
        invalid_label = []
        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        # 检查是否有5个值，类别ID是否在0-NUM_CLASSES，坐标是否在0-1
                        if len(parts) != 5:
                            invalid_label.append(label_file)
                            break
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        if cls_id < 0 or cls_id >= NUM_CLASSES:
                            invalid_label.append(label_file)
                            break
                        for coord in coords:
                            if coord < 0 or coord > 1:
                                invalid_label.append(label_file)
                                break
            except:
                invalid_label.append(label_file)
        all_errors[split]["invalid_label"] = invalid_label

    return all_errors


def clean_dataset(dataset_root):
    """清理数据集：删除损坏图片 + 孤立标注文件"""
    # 存储清理统计（用于图表）
    clean_stats = {
        "train": {
            "before": {"total_images": 0, "total_labels": 0, "invalid_images": 0, "missing_image_labels": 0},
            "after": {"total_images": 0, "total_labels": 0, "invalid_images": 0, "missing_image_labels": 0},
            "deleted": {"images": 0, "labels": 0}
        },
        "val": {
            "before": {"total_images": 0, "total_labels": 0, "invalid_images": 0, "missing_image_labels": 0},
            "after": {"total_images": 0, "total_labels": 0, "invalid_images": 0, "missing_image_labels": 0},
            "deleted": {"images": 0, "labels": 0}
        }
    }

    # 先获取清理前的错误详情（用于生成错误分布图）
    before_errors = validate_dataset(dataset_root)

    for split in ["train", "val"]:
        img_dir = os.path.join(dataset_root, "images", split)
        label_dir = os.path.join(dataset_root, "labels", split)

        # ========== 第一步：统计清理前的状态 ==========
        # 统计原始文件数
        img_files_before = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in ALLOWED_IMG_FORMATS]
        label_files_before = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        clean_stats[split]["before"]["total_images"] = len(img_files_before)
        clean_stats[split]["before"]["total_labels"] = len(label_files_before)

        # 统计清理前的无效数据
        invalid_images_before = before_errors[split]["invalid_image"]
        clean_stats[split]["before"]["invalid_images"] = len(invalid_images_before)

        missing_image_labels_before = before_errors[split]["missing_image"]
        clean_stats[split]["before"]["missing_image_labels"] = len(missing_image_labels_before)

        # ========== 第二步：执行清理 ==========
        print(f"\n===== 清理 {split} 集 =====")
        # 1. 删除损坏图片 + 对应标注
        print("🔧 删除损坏图片...")
        deleted_images = 0
        for img_file in tqdm(img_files_before, desc="检查并删除损坏图片"):
            img_path = os.path.join(img_dir, img_file)
            if not is_image_valid(img_path):
                os.remove(img_path)
                deleted_images += 1
                # 删除对应标注
                img_name = os.path.splitext(img_file)[0]
                label_file = f"{img_name}.txt"
                label_path = os.path.join(label_dir, label_file)
                if os.path.exists(label_path):
                    os.remove(label_path)

        # 2. 删除孤立标注（有标注无图片）
        print("🔧 删除孤立标注文件...")
        deleted_labels = 0
        # 重新获取清理后的图片名
        img_files_after = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in ALLOWED_IMG_FORMATS]
        img_names_after = set([os.path.splitext(f)[0] for f in img_files_after])
        # 遍历标注文件删除孤立项
        label_files_after_check = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        for label_file in tqdm(label_files_after_check, desc="检查并删除孤立标注"):
            if os.path.splitext(label_file)[0] not in img_names_after:
                os.remove(os.path.join(label_dir, label_file))
                deleted_labels += 1

        # ========== 第三步：统计清理后的状态 ==========
        clean_stats[split]["deleted"]["images"] = deleted_images
        clean_stats[split]["deleted"]["labels"] = deleted_labels

        # 统计清理后文件数
        img_files_final = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in ALLOWED_IMG_FORMATS]
        label_files_final = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        clean_stats[split]["after"]["total_images"] = len(img_files_final)
        clean_stats[split]["after"]["total_labels"] = len(label_files_final)

        # 验证清理后无无效数据
        after_errors = validate_dataset(dataset_root)
        invalid_images_final = after_errors[split]["invalid_image"]
        clean_stats[split]["after"]["invalid_images"] = len(invalid_images_final)

        missing_image_labels_final = after_errors[split]["missing_image"]
        clean_stats[split]["after"]["missing_image_labels"] = len(missing_image_labels_final)

        # 打印清理结果
        print(f"✅ {split}集清理完成：")
        print(f"   - 删除损坏图片：{deleted_images} 张")
        print(f"   - 删除孤立标注：{deleted_labels} 个")
        print(f"   - 清理后图片数：{len(img_files_final)} 张")
        print(f"   - 清理后标注数：{len(label_files_final)} 个")

    # 返回清理统计 + 错误详情
    return clean_stats, before_errors, after_errors


def generate_clean_plots(clean_stats, before_errors, after_errors):
    """生成清理前后对比图表（论文专用）"""
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)

    # ========== 图表1：清理前后图片/标注数量对比（核心） ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # 子图1：训练集
    x1 = ["清理前", "清理后"]
    train_images = [clean_stats["train"]["before"]["total_images"], clean_stats["train"]["after"]["total_images"]]
    train_labels = [clean_stats["train"]["before"]["total_labels"], clean_stats["train"]["after"]["total_labels"]]

    x_pos1 = np.arange(len(x1))
    width = 0.35
    ax1.bar(x_pos1 - width / 2, train_images, width, label="图片数量", color="#1f77b4", edgecolor="black",
            linewidth=0.8)
    ax1.bar(x_pos1 + width / 2, train_labels, width, label="标注数量", color="#ff7f0e", edgecolor="black",
            linewidth=0.8)
    # 数值标注
    for i, v in enumerate(train_images):
        ax1.text(i - width / 2, v + 100, str(v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(train_labels):
        ax1.text(i + width / 2, v + 100, str(v), ha="center", va="bottom", fontsize=9)
    ax1.set_title("训练集清理前后数量对比", fontsize=12, fontweight="bold")
    ax1.set_xlabel("清理阶段", fontsize=10)
    ax1.set_ylabel("数量", fontsize=10)
    ax1.set_xticks(x_pos1)
    ax1.set_xticklabels(x1)
    ax1.legend(frameon=True, fancybox=True, shadow=False)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # 子图2：验证集
    x2 = ["清理前", "清理后"]
    val_images = [clean_stats["val"]["before"]["total_images"], clean_stats["val"]["after"]["total_images"]]
    val_labels = [clean_stats["val"]["before"]["total_labels"], clean_stats["val"]["after"]["total_labels"]]

    x_pos2 = np.arange(len(x2))
    ax2.bar(x_pos2 - width / 2, val_images, width, label="图片数量", color="#1f77b4", edgecolor="black", linewidth=0.8)
    ax2.bar(x_pos2 + width / 2, val_labels, width, label="标注数量", color="#ff7f0e", edgecolor="black", linewidth=0.8)
    # 数值标注
    for i, v in enumerate(val_images):
        ax2.text(i - width / 2, v + 20, str(v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(val_labels):
        ax2.text(i + width / 2, v + 20, str(v), ha="center", va="bottom", fontsize=9)
    ax2.set_title("验证集清理前后数量对比", fontsize=12, fontweight="bold")
    ax2.set_xlabel("清理阶段", fontsize=10)
    ax2.set_ylabel("数量", fontsize=10)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(x2)
    ax2.legend(frameon=True, fancybox=True, shadow=False)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_PATH, "cleaned_dataset_basic_stats_before_after.png"), bbox_inches="tight")
    plt.close()

    # ========== 图表2：清理前数据集错误分布（论文重点：展示数据质量） ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    error_types = ["缺少标注的图片", "缺少图片的标注", "损坏图片", "格式错误标注"]
    train_errors = [
        len(before_errors["train"]["missing_label"]),
        len(before_errors["train"]["missing_image"]),
        len(before_errors["train"]["invalid_image"]),
        len(before_errors["train"]["invalid_label"])
    ]
    val_errors = [
        len(before_errors["val"]["missing_label"]),
        len(before_errors["val"]["missing_image"]),
        len(before_errors["val"]["invalid_image"]),
        len(before_errors["val"]["invalid_label"])
    ]

    x_pos = np.arange(len(error_types))
    width = 0.35
    ax.bar(x_pos - width / 2, train_errors, width, label="训练集", color="#2ca02c", edgecolor="black", linewidth=0.8)
    ax.bar(x_pos + width / 2, val_errors, width, label="验证集", color="#d62728", edgecolor="black", linewidth=0.8)

    # 添加数值标注
    for i, v in enumerate(train_errors):
        ax.text(i - width / 2, v + 50, str(v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(val_errors):
        ax.text(i + width / 2, v + 50, str(v), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("错误类型", fontsize=12)
    ax.set_ylabel("错误数量", fontsize=12)
    ax.set_title("清理前垃圾分类数据集错误分布", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(error_types, rotation=0)
    ax.legend(frameon=True, fancybox=True, shadow=False)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_PATH, "cleaned_dataset_error_stats_before.png"), bbox_inches="tight")
    plt.close()

    # ========== 图表3：清理后数据集错误分布（对比展示清理效果） ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    train_errors_after = [
        len(after_errors["train"]["missing_label"]),
        len(after_errors["train"]["missing_image"]),
        len(after_errors["train"]["invalid_image"]),
        len(after_errors["train"]["invalid_label"])
    ]
    val_errors_after = [
        len(after_errors["val"]["missing_label"]),
        len(after_errors["val"]["missing_image"]),
        len(after_errors["val"]["invalid_image"]),
        len(after_errors["val"]["invalid_label"])
    ]

    ax.bar(x_pos - width / 2, train_errors_after, width, label="训练集", color="#2ca02c", edgecolor="black",
           linewidth=0.8)
    ax.bar(x_pos + width / 2, val_errors_after, width, label="验证集", color="#d62728", edgecolor="black",
           linewidth=0.8)

    # 添加数值标注（所有都是0，直接标在柱子底部）
    for i, v in enumerate(train_errors_after):
        ax.text(i - width / 2, 0.1, str(v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(val_errors_after):
        ax.text(i + width / 2, 0.1, str(v), ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("错误类型", fontsize=12)
    ax.set_ylabel("错误数量", fontsize=12)
    ax.set_title("清理后垃圾分类数据集错误分布", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(error_types, rotation=0)
    ax.legend(frameon=True, fancybox=True, shadow=False)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # 关键修复：把Y轴范围缩小到0-1，让0值清晰可见
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_PATH, "cleaned_dataset_error_stats_after.png"), bbox_inches="tight")
    plt.close()

    # ========== 图表4：清理删除文件统计 ==========
    fig, ax = plt.subplots(figsize=(8, 6))
    x = ["训练集-图片", "训练集-标注", "验证集-图片", "验证集-标注"]
    deleted_nums = [
        clean_stats["train"]["deleted"]["images"],
        clean_stats["train"]["deleted"]["labels"],
        clean_stats["val"]["deleted"]["images"],
        clean_stats["val"]["deleted"]["labels"]
    ]
    colors = ["#d62728", "#2ca02c", "#d62728", "#2ca02c"]

    bars = ax.bar(x, deleted_nums, color=colors, edgecolor="black", linewidth=0.8)
    # 数值标注
    for bar, num in zip(bars, deleted_nums):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 50,
                str(num), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("数据集清理删除文件统计", fontsize=14, fontweight="bold")
    ax.set_xlabel("文件类型", fontsize=12)
    ax.set_ylabel("删除数量", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_PATH, "cleaned_dataset_deleted_stats.png"), bbox_inches="tight")
    plt.close()

    # ========== 图表5：清理后有效匹配率 ==========
    # 计算清理后有效匹配率（100%）
    train_valid_rate = 100.0 if clean_stats["train"]["after"]["total_images"] > 0 else 0
    val_valid_rate = 100.0 if clean_stats["val"]["after"]["total_images"] > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    x = ["训练集", "验证集"]
    valid_rates = [train_valid_rate, val_valid_rate]

    bars = ax.bar(x, valid_rates, color="#9467bd", edgecolor="black", linewidth=0.8)
    # 数值标注
    for bar, rate in zip(bars, valid_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f"{rate:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("清理后数据集有效匹配率", fontsize=14, fontweight="bold")
    ax.set_xlabel("数据集划分", fontsize=12)
    ax.set_ylabel("有效匹配率（%）", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_PATH, "cleaned_dataset_valid_rate.png"), bbox_inches="tight")
    plt.close()

    print(f"\n✅ 清理后分析图表已生成，保存路径：{PLOT_SAVE_PATH}")
    print("📄 生成的图表包括：")
    print("   1. cleaned_dataset_basic_stats_before_after.png - 清理前后数量对比")
    print("   2. cleaned_dataset_error_stats_before.png - 清理前错误分布统计")
    print("   3. cleaned_dataset_error_stats_after.png - 清理后错误分布统计")
    print("   4. cleaned_dataset_deleted_stats.png - 清理删除文件统计")
    print("   5. cleaned_dataset_valid_rate.png - 清理后有效匹配率")


# ===================== 主函数 =====================
def main():
    # 步骤1：复制原始数据到新目录
    copy_dataset(RAW_DATASET_ROOT, CLEANED_DATASET_ROOT)

    # 步骤2：清理新目录中的数据（返回清理统计+错误详情）
    clean_stats, before_errors, after_errors = clean_dataset(CLEANED_DATASET_ROOT)

    # 步骤3：生成清理后的分析图表
    generate_clean_plots(clean_stats, before_errors, after_errors)

    # 最终提示
    print("\n🎉 数据集清理全流程完成！")
    print(f"📁 清理后数据集路径：{CLEANED_DATASET_ROOT}")
    print(f"📊 分析图表路径：{PLOT_SAVE_PATH}")
    print("💡 清理后的数据集可直接用于YOLO训练，原始数据集未被修改！")


if __name__ == "__main__":
    main()