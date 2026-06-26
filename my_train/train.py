from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo12s.pt")
    model.train(
        data = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\cfg\datasets\data.yaml",
        epochs = 100,
        batch = 16,
        imgsz = 640,
        workers = 0,
        patience = 15,
        mosaic = 1.0,
        mixup = 0.1,
        degrees = 10,
        scale = 0.1,
        fliplr = 0.5,
        device = "0",
        project = "results",
        name = "yolo12s",
    )