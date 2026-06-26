import os
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ===================== 核心配置项 =====================
DATASET_ROOT = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\balanced_garbage_datasets"
PLOT_SAVE_PATH = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\balanced_garbage_datasets_analysis_plots"

CLASS_INFO = [
  {"id": 0, "name_en": "FastFoodBox", "name_cn": "快餐盒", "category": "OtherGarbage"},
  {"id": 1, "name_en": "SoiledPlastic", "name_cn": "污损塑料", "category": "OtherGarbage"},
  {"id": 2, "name_en": "Cigarette", "name_cn": "烟头", "category": "OtherGarbage"},
  {"id": 3, "name_en": "Toothpick", "name_cn": "牙签", "category": "OtherGarbage"},
  {"id": 4, "name_en": "Flowerpot", "name_cn": "花盆", "category": "OtherGarbage"},
  {"id": 5, "name_en": "BambooChopstics", "name_cn": "竹筷", "category": "OtherGarbage"},
  {"id": 6, "name_en": "Meal", "name_cn": "剩饭剩菜", "category": "KitchenWaste"},
  {"id": 7, "name_en": "Bone", "name_cn": "骨头", "category": "KitchenWaste"},
  {"id": 8, "name_en": "FruitPeel", "name_cn": "水果皮", "category": "KitchenWaste"},
  {"id": 9, "name_en": "Pulp", "name_cn": "纸浆", "category": "KitchenWaste"},
  {"id": 10, "name_en": "Tea", "name_cn": "茶叶", "category": "KitchenWaste"},
  {"id": 11, "name_en": "Vegetable", "name_cn": "蔬菜", "category": "KitchenWaste"},
  {"id": 12, "name_en": "Eggshell", "name_cn": "蛋壳", "category": "KitchenWaste"},
  {"id": 13, "name_en": "FishBone", "name_cn": "鱼骨", "category": "KitchenWaste"},
  {"id": 14, "name_en": "Powerbank", "name_cn": "充电宝", "category": "Recyclables"},
  {"id": 15, "name_en": "Bag", "name_cn": "包", "category": "Recyclables"},
  {"id": 16, "name_en": "CosmeticBottles", "name_cn": "化妆品瓶", "category": "Recyclables"},
  {"id": 17, "name_en": "Toys", "name_cn": "玩具", "category": "Recyclables"},
  {"id": 18, "name_en": "PlasticBowl", "name_cn": "塑料碗", "category": "Recyclables"},
  {"id": 19, "name_en": "PlasticHanger", "name_cn": "塑料衣架", "category": "Recyclables"},
  {"id": 20, "name_en": "PaperBags", "name_cn": "纸袋", "category": "Recyclables"},
  {"id": 21, "name_en": "PlugWire", "name_cn": "插头电线", "category": "Recyclables"},
  {"id": 22, "name_en": "OldClothes", "name_cn": "旧衣物", "category": "Recyclables"},
  {"id": 23, "name_en": "Can", "name_cn": "易拉罐", "category": "Recyclables"},
  {"id": 24, "name_en": "Pillow", "name_cn": "枕头", "category": "Recyclables"},
  {"id": 25, "name_en": "PlushToys", "name_cn": "毛绒玩具", "category": "Recyclables"},
  {"id": 26, "name_en": "ShampooBottle", "name_cn": "洗发水瓶", "category": "Recyclables"},
  {"id": 27, "name_en": "GlassCup", "name_cn": "玻璃杯", "category": "Recyclables"},
  {"id": 28, "name_en": "Shoes", "name_cn": "鞋子", "category": "Recyclables"},
  {"id": 29, "name_en": "Anvil", "name_cn": "铁砧", "category": "Recyclables"},
  {"id": 30, "name_en": "Cardboard", "name_cn": "纸板箱", "category": "Recyclables"},
  {"id": 31, "name_en": "SeasoningBottle", "name_cn": "调味品瓶", "category": "Recyclables"},
  {"id": 32, "name_en": "Bottle", "name_cn": "酒瓶", "category": "Recyclables"},
  {"id": 33, "name_en": "MetalFoodCans", "name_cn": "金属食品罐", "category": "Recyclables"},
  {"id": 34, "name_en": "Pot", "name_cn": "锅", "category": "Recyclables"},
  {"id": 35, "name_en": "EdibleOilBarrel", "name_cn": "食用油桶", "category": "Recyclables"},
  {"id": 36, "name_en": "DrinkBottle", "name_cn": "饮料瓶", "category": "Recyclables"},
  {"id": 37, "name_en": "DryBattery", "name_cn": "干电池", "category": "HazardousWaste"},
  {"id": 38, "name_en": "Ointment", "name_cn": "药膏", "category": "HazardousWaste"},
  {"id": 39, "name_en": "ExpiredDrugs", "name_cn": "过期药品", "category": "HazardousWaste"}
]

plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

# ===================== 统计类别数量 =====================
def count_class_samples(dataset_root):
    class_counts = defaultdict(int)
    split_counts = {"train": defaultdict(int), "val": defaultdict(int)}

    for split in ["train", "val"]:
        label_dir = os.path.join(dataset_root, "labels", split)
        if not os.path.exists(label_dir):
            continue

        for label_file in os.listdir(label_dir):
            if not label_file.endswith(".txt"):
                continue
            label_path = os.path.join(label_dir, label_file)

            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line: continue
                        cls_id = int(line.split()[0])
                        class_counts[cls_id] += 1
                        split_counts[split][cls_id] += 1
            except:
                continue

    return class_counts, split_counts

# ===================== 画训练集 vs 验证集图 =====================
def plot_train_val_distribution(class_counts, split_counts):
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
    id_to_name = {item["id"]: item["name_en"] for item in CLASS_INFO}
    class_ids = sorted(class_counts.keys())
    class_names = [id_to_name[cid] for cid in class_ids]

    train_counts = [split_counts["train"].get(cid, 0) for cid in class_ids]
    val_counts = [split_counts["val"].get(cid, 0) for cid in class_ids]

    fig, ax = plt.subplots(figsize=(18, 8))
    width = 0.35
    x = list(range(len(class_ids)))

    bars1 = ax.bar([i - width/2 for i in x], train_counts, width, label="Train", color="#2ca02c")
    bars2 = ax.bar([i + width/2 for i in x], val_counts, width, label="Val", color="#ff7f0e")

    # 训练集数字
    for bar, num in zip(bars1, train_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{num}', ha='center', va='bottom', fontsize=5)
    # 验证集数字
    for bar, num in zip(bars2, val_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{num}', ha='center', va='bottom', fontsize=5)

    ax.set_title("修复后类别分布图", fontsize=14, weight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("数量")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(PLOT_SAVE_PATH, "balanced_classes_count_stats.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# ===================== 生成专业表格图 =====================
def plot_class_table(class_counts, split_counts):
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
    id_to_name = {item["id"]: item["name_en"] for item in CLASS_INFO}
    id_to_cn = {item["id"]: item["name_cn"] for item in CLASS_INFO}
    class_ids = sorted(class_counts.keys())

    # 构造表格数据
    headers = ["ID", "英文名称", "中文名称", "训练集", "验证集", "总数"]
    rows = []
    for cid in class_ids:
        t = split_counts["train"].get(cid, 0)
        v = split_counts["val"].get(cid, 0)
        total = t + v
        rows.append([
            str(cid),
            id_to_name[cid],
            id_to_cn[cid],
            str(t),
            str(v),
            str(total)
        ])

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=[0.05, 0.28, 0.12, 0.12, 0.12, 0.12]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    save_path = os.path.join(PLOT_SAVE_PATH, "balanced_classes_count_table.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("✅ 表格图已保存：", save_path)

# ===================== 主函数 =====================
def main():
    class_counts, split_counts = count_class_samples(DATASET_ROOT)
    plot_train_val_distribution(class_counts, split_counts)
    plot_class_table(class_counts, split_counts)
    print("\n🎉 全部生成完成！")

if __name__ == "__main__":
    main()