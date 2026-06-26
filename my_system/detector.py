import cv2
import time
import gc
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from ultralytics import YOLO
from config import CLASS_NAMES

class DetectorThread(QThread):
    frame_ready = pyqtSignal(QImage, list, float)

    def __init__(self, mode, source, model: YOLO, iou=0.5):
        super().__init__()
        self.mode = mode
        self.source = source
        self.model = model
        self.run_flag = True
        self.iou = iou

    def run(self):
        try:
            if self.mode == "image":
                img = cv2.imread(self.source)
                if img is None:
                    return
                start = time.time()
                res = self.model(img, iou=self.iou, verbose=False)[0]
                infer_time = time.time() - start
                out_img = res.plot()
                dets = self._parse(res)
                self.frame_ready.emit(self._cvt(out_img), dets, infer_time)

            else:
                cap = cv2.VideoCapture(0 if self.mode == "camera" else self.source)

                # ====================== 视频速度控制 ======================
                frame_interval = 0  # 摄像头不限速
                if self.mode != "camera":
                    # 获取视频原帧率
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fps = max(10, min(fps, 30))  # 限制在合理范围
                    frame_interval = 1.0 / fps
                # ==========================================================

                while self.run_flag:
                    ret, f = cap.read()
                    if not ret:
                        break

                    frame = f.copy()
                    start = time.time()

                    res = self.model(frame, iou=self.iou, verbose=False)[0]
                    infer_time = time.time() - start
                    out_img = res.plot()
                    dets = self._parse(res)
                    self.frame_ready.emit(self._cvt(out_img), dets, infer_time)

                    # ====================== 只给视频加延迟 ======================
                    if self.mode != "camera":
                        used = time.time() - start
                        wait = max(0.0, frame_interval - used)
                        time.sleep(wait)

                cap.release()
        except Exception as e:
            print("线程安全退出")
        finally:
            gc.collect()

    def _parse(self, res):
        arr = []
        for box in res.boxes:
            try:
                c = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else "未知"
                arr.append({"name": name, "conf": conf, "xyxy": (x1, y1, x2, y2)})
            except:
                continue
        return arr

    def _cvt(self, im):
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return qimg.copy()

    def stop(self):
        self.run_flag = False
        self.wait()