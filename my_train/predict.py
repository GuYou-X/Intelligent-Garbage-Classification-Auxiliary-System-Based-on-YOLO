from ultralytics import YOLO
import os
import cv2
import numpy as np

if __name__ == "__main__":
    model = YOLO(r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\results\yolo12s\weights\best.pt")
    model.predict(
        source = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_system\test\images2",
        imgsz = 640,
        conf = 0.25,
        save = True,
        project = "predict",
        name = "yolo12s",
        exist_ok = True
    )

    ######################## 自动生成网格大图（你要的效果） ########################
    # 配置
    result_dir = r"predict/yolo12s"  # 你的输出目录
    rows = 4  # 几行
    cols = 4  # 几列
    img_size = (320, 320)

    # 读取所有图片
    images = []
    for f in sorted(os.listdir(result_dir)):
        if f.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(result_dir, f))
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)

    # 只取前 N 张
    images = images[:rows * cols]

    # 补全空白
    while len(images) < rows * cols:
        images.append(np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8))

    # 拼接成网格
    grid = []
    for i in range(rows):
        row = np.hstack(images[i * cols: (i + 1) * cols])
        grid.append(row)
    final = np.vstack(grid)

    # 保存总览图
    cv2.imwrite(os.path.join(result_dir, "test_summary_grid.png"), final)
    print("✅ 网格大图已生成：", os.path.join(result_dir, "test_summary_grid.png"))