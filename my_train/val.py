from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\results\yolo12s\weights\best.pt")
    model.val(
        project = "val",
        name = "yolo12s",
        exist_ok = True
    )