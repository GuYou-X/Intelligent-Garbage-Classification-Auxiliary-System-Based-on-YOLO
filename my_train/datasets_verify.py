import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')  # 忽略matplotlib字体警告

# ===================== 配置项 =====================
# 原始数据集根路径
DATASET_ROOT = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\garbage_datasets\datasets"
# 图表保存根路径（改为my_train目录）
MY_TRAIN_ROOT = os.path.dirname(os.path.dirname(DATASET_ROOT))  # 自动定位到my_train目录
# 垃圾分类类别数
NUM_CLASSES = 40
# 允许的图片格式
ALLOWED_IMG_FORMATS = [".jpg"]
# 图表保存路径（my_train/dataset_analysis_plots）
PLOT_SAVE_PATH = os.path.join(MY_TRAIN_ROOT, "garbage_datasets_analysis_plots")

# 设置matplotlib字体
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]  # 中文+英文兼容
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.dpi"] = 300  # 高清图片（300DPI，符合论文要求）


# ===================== 核心函数 =====================
def check_image_valid(img_path):
    """检查单张图片是否损坏、可读取"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, "图片无法读取（损坏/格式错误）"
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return False, "图片尺寸异常（宽/高为0）"
        return True, "图片正常"
    except Exception as e:
        return False, f"图片读取报错：{str(e)}"


def check_label_yolo_format(label_path, num_classes):
    """检查单个标注文件是否符合YOLO格式"""
    errors = []
    if not os.path.exists(label_path):
        errors.append("标注文件不存在")
        return False, errors

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(f"第{line_idx}行：格式错误（需5个值，实际{len(parts)}个）")
                continue

            try:
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= num_classes:
                    errors.append(f"第{line_idx}行：类别ID{cls_id}超出范围（0~{num_classes - 1}）")
            except ValueError:
                errors.append(f"第{line_idx}行：类别ID{parts[0]}不是整数")

            for i, coord in enumerate(parts[1:], 1):
                try:
                    val = float(coord)
                    if val < 0 or val > 1:
                        errors.append(f"第{line_idx}行：坐标值{coord}超出0-1范围")
                except ValueError:
                    errors.append(f"第{line_idx}行：坐标{coord}不是浮点数")

    except Exception as e:
        errors.append(f"标注文件读取报错：{str(e)}")

    return len(errors) == 0, errors


def verify_dataset():
    """整体校验数据集 + 收集统计数据"""
    # 存储错误信息 + 统计数据
    all_errors = {
        "train": {"missing_label": [], "missing_image": [], "invalid_image": [], "invalid_label": []},
        "val": {"missing_label": [], "missing_image": [], "invalid_image": [], "invalid_label": []}
    }
    # 新增：统计基础数据（用于图表）
    stats = {
        "train": {"total_images": 0, "total_labels": 0},
        "val": {"total_images": 0, "total_labels": 0}
    }

    # 遍历train/val集
    for split in ["train", "val"]:
        img_dir = os.path.join(DATASET_ROOT, "images", split)
        label_dir = os.path.join(DATASET_ROOT, "labels", split)

        if not os.path.exists(img_dir):
            print(f"⚠️  警告：{img_dir} 文件夹不存在！")
            continue
        if not os.path.exists(label_dir):
            print(f"⚠️  警告：{label_dir} 文件夹不存在！")
            continue

        # 1. 获取文件列表并统计总数
        img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in ALLOWED_IMG_FORMATS]
        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        stats[split]["total_images"] = len(img_files)
        stats[split]["total_labels"] = len(label_files)

        # 2. 校验图片→标注
        print(f"\n===== 校验 {split} 集：图片→标注对应关系 =====")
        for img_file in tqdm(img_files, desc="检查图片对应标注"):
            img_name = os.path.splitext(img_file)[0]
            label_file = f"{img_name}.txt"
            img_path = os.path.join(img_dir, img_file)

            # 检查图片有效性
            is_img_valid, img_err = check_image_valid(img_path)
            if not is_img_valid:
                all_errors[split]["invalid_image"].append(f"{img_file}：{img_err}")

            # 检查标注是否存在
            if label_file not in label_files:
                all_errors[split]["missing_label"].append(img_file)
                continue

            # 检查标注格式
            label_path = os.path.join(label_dir, label_file)
            is_label_valid, label_errs = check_label_yolo_format(label_path, NUM_CLASSES)
            if not is_label_valid:
                all_errors[split]["invalid_label"].append(f"{label_file}：{'; '.join(label_errs)}")

        # 3. 校验标注→图片
        print(f"\n===== 校验 {split} 集：标注→图片对应关系 =====")
        for label_file in tqdm(label_files, desc="检查标注对应图片"):
            label_name = os.path.splitext(label_file)[0]
            img_exists = False
            for fmt in ALLOWED_IMG_FORMATS:
                img_file = f"{label_name}{fmt}"
                if img_file in img_files:
                    img_exists = True
                    break
            if not img_exists:
                all_errors[split]["missing_image"].append(label_file)

    # 4. 输出校验报告
    print("\n" + "=" * 50)
    print("📊 数据集校验报告")
    print("=" * 50)
    for split in ["train", "val"]:
        print(f"\n【{split.upper()} 集】")
        err = all_errors[split]
        print(f"📈 基础统计：图片总数={stats[split]['total_images']}，标注总数={stats[split]['total_labels']}")

        if err["missing_label"]:
            print(f"❌ 缺少标注的图片（共{len(err['missing_label'])}个）：")
            for f in err["missing_label"][:10]:
                print(f"  - {f}")
            if len(err["missing_label"]) > 10:
                print(f"  ... 还有{len(err['missing_label']) - 10}个未显示")

        if err["missing_image"]:
            print(f"❌ 缺少图片的标注（共{len(err['missing_image'])}个）：")
            for f in err["missing_image"][:10]:
                print(f"  - {f}")
            if len(err["missing_image"]) > 10:
                print(f"  ... 还有{len(err['missing_image']) - 10}个未显示")

        if err["invalid_image"]:
            print(f"❌ 损坏/无效的图片（共{len(err['invalid_image'])}个）：")
            for f in err["invalid_image"][:10]:
                print(f"  - {f}")
            if len(err["invalid_image"]) > 10:
                print(f"  ... 还有{len(err['invalid_image']) - 10}个未显示")

        if err["invalid_label"]:
            print(f"❌ 格式错误的标注（共{len(err['invalid_label'])}个）：")
            for f in err["invalid_label"][:10]:
                print(f"  - {f}")
            if len(err["invalid_label"]) > 10:
                print(f"  ... 还有{len(err['invalid_label']) - 10}个未显示")

        if not any(err.values()):
            print(f"✅ {split} 集所有校验项通过！")

    # 5. 生成论文用图表
    generate_paper_plots(all_errors, stats)
    return all_errors, stats


def generate_paper_plots(all_errors, stats):
    # 创建图表保存目录（my_train/dataset_analysis_plots）
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)

    # ========== 图表1：数据集基础统计（训练/验证集图片/标注数量对比） ==========
    fig, ax = plt.subplots(figsize=(8, 5))
    x = ["训练集", "验证集"]
    images = [stats["train"]["total_images"], stats["val"]["total_images"]]
    labels = [stats["train"]["total_labels"], stats["val"]["total_labels"]]

    x_pos = np.arange(len(x))
    width = 0.35
    ax.bar(x_pos - width / 2, images, width, label="图片数量", color="#1f77b4", edgecolor="black", linewidth=0.8)
    ax.bar(x_pos + width / 2, labels, width, label="标注数量", color="#ff7f0e", edgecolor="black", linewidth=0.8)

    # 添加数值标注
    for i, v in enumerate(images):
        ax.text(i - width / 2, v + 100, str(v), ha="center", va="bottom", fontsize=10)
    for i, v in enumerate(labels):
        ax.text(i + width / 2, v + 100, str(v), ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("数据集划分", fontsize=12)
    ax.set_ylabel("数量", fontsize=12)
    ax.set_title("垃圾分类数据集基础统计", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.legend(frameon=True, fancybox=True, shadow=False)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_PATH, "dataset_basic_stats.png"), bbox_inches="tight")
    plt.close()

    # ========== 图表2：数据集错误分布 ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    error_types = ["缺少标注的图片", "缺少图片的标注", "损坏图片", "格式错误标注"]
    train_errors = [
        len(all_errors["train"]["missing_label"]),
        len(all_errors["train"]["missing_image"]),
        len(all_errors["train"]["invalid_image"]),
        len(all_errors["train"]["invalid_label"])
    ]
    val_errors = [
        len(all_errors["val"]["missing_label"]),
        len(all_errors["val"]["missing_image"]),
        len(all_errors["val"]["invalid_image"]),
        len(all_errors["val"]["invalid_label"])
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
    ax.set_title("垃圾分类数据集错误分布", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(error_types, rotation=0)
    ax.legend(frameon=True, fancybox=True, shadow=False)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_PATH, "dataset_error_stats.png"), bbox_inches="tight")
    plt.close()

    # ========== 图表3：数据集有效率计算 ==========
    # 有效图片数 = 总图片数 - 损坏图片数
    train_valid_images = stats["train"]["total_images"] - len(all_errors["train"]["invalid_image"])
    val_valid_images = stats["val"]["total_images"] - len(all_errors["val"]["invalid_image"])
    # 有效标注匹配数 = 有效图片数 - 缺少标注的图片数
    train_matched = train_valid_images - len(all_errors["train"]["missing_label"])
    val_matched = val_valid_images - len(all_errors["val"]["missing_label"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = ["训练集", "验证集"]
    valid_rates = [
        train_matched / stats["train"]["total_images"] * 100,
        val_matched / stats["val"]["total_images"] * 100
    ]

    bars = ax.bar(x, valid_rates, color="#9467bd", edgecolor="black", linewidth=0.8)
    # 添加百分比标注
    for bar, rate in zip(bars, valid_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f"{rate:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("数据集划分", fontsize=12)
    ax.set_ylabel("有效匹配率（%）", fontsize=12)
    ax.set_title("垃圾分类数据集图片-标注有效匹配率", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)  # 百分比从0到105，留出标注空间
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_PATH, "dataset_valid_rate.png"), bbox_inches="tight")
    plt.close()

    print(f"\n✅ 论文用图表已生成，保存路径：{PLOT_SAVE_PATH}")
    print("📄 生成的图表包括：")
    print("   1. dataset_basic_stats.png - 数据集基础统计（图片/标注数量）")
    print("   2. dataset_error_stats.png - 数据集错误分布统计")
    print("   3. dataset_valid_rate.png - 图片-标注有效匹配率")


# ===================== 主函数 =====================
if __name__ == "__main__":
    # 执行校验 + 生成图表
    verify_dataset()