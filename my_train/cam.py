from ultralytics import YOLO
import cv2

model = YOLO(r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_train\results\test1\weights\best.pt")

results = model(
    source = 0,
    stream = True,
)

for result in results:
    plotted = result.plot()
    cv2.imshow("YOLO Inference",plotted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()