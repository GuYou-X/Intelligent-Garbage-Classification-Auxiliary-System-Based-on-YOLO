import sys
import os
import gc
import glob
import time
import csv
import cv2
import win32com.client
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ultralytics import YOLO
from detector import DetectorThread
from config import GARBAGE_TYPE_MAP, CLASS_NAMES
from openai import OpenAI

# ====================== 语音线程 ======================
class VoiceWorker(QThread):
    def __init__(self):
        super().__init__()
        self.queue = []
        self.speaker = win32com.client.Dispatch("SAPI.SpVoice")

    def add(self, text):
        self.queue.append(text)

    def run(self):
        while True:
            if self.queue:
                txt = self.queue.pop(0)
                try:
                    self.speaker.Speak(txt)
                except:
                    pass
            QThread.msleep(50)

# ====================== AI 总结线程 ======================
class AISummaryThread(QThread):
    reply_ready = pyqtSignal(str)

    def __init__(self, client, text):
        super().__init__()
        self.client = client
        self.text = text

    def run(self):
        try:
            # 提示词
            prompt = (
                "你是专业的垃圾分类总结助手，请用一句简洁自然的话总结这些垃圾分类，要求准确指出垃圾类别以及对应投放垃圾桶。"
                f"待总结内容：{self.text}"
            )
            response = self.client.chat.completions.create(
                model="qwen/qwen3-vl-4b",
                messages=[{"role": "user", "content": prompt.strip()}],
                max_tokens=256,
                temperature=0.2,
                stream=False
            )
            res = response.choices[0].message.content.strip()
            self.reply_ready.emit(res)
        except Exception as e:
            # AI异常就友好提示，不显示杂乱原文
            self.reply_ready.emit("AI总结暂不可用，已完成本次垃圾分类检测。")

# ====================== 文件夹批量检测线程 ======================
class FolderProcessThread(QThread):
    progress = pyqtSignal(str, QImage, list, float)
    finished = pyqtSignal()

    def __init__(self, folder, model, iou):
        super().__init__()
        self.folder = folder
        self.model = model
        self.iou = iou
        self.run_flag = True

    def run(self):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self.folder, ext)))
            files.extend(glob.glob(os.path.join(self.folder, ext.upper())))

        for fpath in files:
            if not self.run_flag:
                break
            try:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                start = time.time()
                res = self.model(img, iou=self.iou, verbose=False)[0]
                t = time.time() - start
                out_img = res.plot()
                dets = []
                for box in res.boxes:
                    try:
                        c = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else "未知"
                        dets.append({"name": name, "conf": conf, "xyxy": (x1, y1, x2, y2)})
                    except:
                        continue

                rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
                self.progress.emit(os.path.basename(fpath), qimg, dets, t)
                time.sleep(0.3)
            except Exception as e:
                print(e)
        self.finished.emit()

    def stop(self):
        self.run_flag = False
        self.wait()

# ====================== 主界面 ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLO的智能垃圾分类辅助系统")
        self.setGeometry(100, 100, 1700, 1100)
        self.setStyleSheet("""
            QMainWindow {
                border-image: url(./background.png) 0 0 0 0 stretch;
                background-color: rgba(240,247,255,0.4);
            }
            QGroupBox {
                font-family: "Microsoft YaHei";
                font-size: 35px;
                font-weight: bold;
                color: #2D3748;
                border: none;
                border-radius: 20px;
                margin-top: 16px;
                padding-top: 16px;
                background-color: rgba(255, 255, 255, 0.4);
                border: 1px solid rgba(255,255,255,0.3);
                box-shadow: 0 8px 20px rgba(0, 100, 200, 0.08);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 10px;
                color: #2B6CB0;
            }
            QPushButton {
                font-family: "Microsoft YaHei";
                font-size: 30px;
                font-weight: bold;
                height: 60px;
                border-radius: 14px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4299E1,
                    stop:1 #3182CE
                );
                color: white;
                border: none;
                padding: 0 20px;
                box-shadow: 0 4px 10px rgba(66, 153, 225, 0.2);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #63B3ED,
                    stop:1 #4299E1
                );
                box-shadow: 0 6px 14px rgba(66, 153, 225, 0.3);
            }
            QPushButton:pressed {
                background: #2C5282;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
            }
            QLabel {
                font-family: "Microsoft YaHei";
                font-size: 30px;
                color: #4A5568;
                background: transparent;
            }
            QLineEdit, QComboBox {
                font-family: "Microsoft YaHei";
                font-size: 30px;
                height: 50px;
                border-radius: 12px;
                border: 1px solid #E2E8F0;
                background-color: rgba(255,255,255,0.5);
                padding: 0 14px;
                color:#2D3748;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #90CDF4;
                background-color: rgba(255,255,255,0.9);
                box-shadow: 0 0 8px rgba(66, 153, 225, 0.15);
            }
            QTextEdit {
                font-family: "Microsoft YaHei";
                font-size: 30px;
                border-radius: 14px;
                border: 1px solid #E2E8F0;
                background-color: rgba(255,255,255,0.4);
                padding:14px;
                color:#2D3748;
            }
            QTableWidget {
                font-family: "Microsoft YaHei";
                font-size: 25px;
                gridline-color: #E2E8F0;
                border:none;
                background-color: rgba(255,255,255,0.35);
                border-radius:16px;
                box-shadow: 0 6px 16px rgba(0, 100, 200, 0.06);
            }
            QHeaderView::section {
                font-family: "Microsoft YaHei";
                font-weight:bold;
                font-size: 25px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #EBF8FF,
                    stop:1 #E6FFFA
                );
                border:none;
                padding: 14px;
                color: #2C5282;
            }
            QTableWidget::item {
                padding: 12px;
                border-bottom:1px solid #F7FAFC;
                background: transparent;
            }
            #img_label {
                border: none;
                border-radius: 20px;
                background-color: rgba(255,255,255,0.4);
                padding: 4px;
                box-shadow: 0 10px 25px rgba(0,100,200,0.08);
            }
        """)

        self.thread = None
        self.folder_thread = None
        self.model = None
        self.is_camera_running = False
        self.last_items = set()
        self.voice_enabled = True

        self.all_detected_objects = set()

        self.voice_worker = VoiceWorker()
        self.voice_worker.start()

        try:
            self.ai_client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
            self.ai_available = True
        except:
            self.ai_available = False

        self.ai_thread = None
        self.initUI()

    def initUI(self):
        c = QWidget()
        self.setCentralWidget(c)
        main_layout = QVBoxLayout(c)
        main_layout.setSpacing(28)
        main_layout.setContentsMargins(50, 50, 50, 50)

        # 标题区域
        title_layout = QVBoxLayout()
        title = QLabel("♻️ 基于YOLO的智能垃圾分类辅助系统")
        title.setFont(QFont("Microsoft YaHei", 60, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2C5282; margin-bottom: 10px;")

        sub_title = QLabel("AI视觉识别 | 实时检测 | 批量分析 | 智能总结 | 语音播报")
        sub_title.setFont(QFont("Microsoft YaHei", 40))
        sub_title.setAlignment(Qt.AlignCenter)
        sub_title.setStyleSheet("color: #718096;")

        title_layout.addWidget(title)
        title_layout.addWidget(sub_title)
        main_layout.addLayout(title_layout)

        # 中间三栏布局
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(28)

        # 左侧面板
        left_group = QGroupBox("📊 检测信息")
        left_group.setFixedWidth(500)
        left_layout = QVBoxLayout(left_group)
        left_layout.setSpacing(5)

        self.time_label = QLabel("⏱️ 推理用时：0.00s")
        self.num_label = QLabel("🎯 检测目标：0 个")
        self.type_label = QLabel("👁️ 目标类型：-")
        self.conf_label = QLabel("📈 置信度：0.00%")
        self.category_label = QLabel("🗑️ 垃圾类别：-")

        for w in [self.time_label, self.num_label, self.type_label, self.conf_label, self.category_label]:
            w.setFont(QFont("Microsoft YaHei", 40, QFont.Bold))
            w.setStyleSheet("margin-top: 4px; margin-bottom: 4px;")

        ai_label = QLabel("🧠 AI 智能总结")
        ai_label.setFont(QFont("Microsoft YaHei",40,QFont.Bold))
        self.ai_summary_edit = QTextEdit()
        self.ai_summary_edit.setReadOnly(True)
        self.ai_summary_edit.setMaximumHeight(180)

        left_layout.addWidget(self.time_label)
        left_layout.addWidget(self.num_label)
        left_layout.addWidget(self.type_label)
        left_layout.addWidget(self.conf_label)
        left_layout.addWidget(self.category_label)
        left_layout.addSpacing(10)
        left_layout.addWidget(ai_label)
        left_layout.addWidget(self.ai_summary_edit)
        mid_layout.addWidget(left_group)

        # 中间显示区域
        self.img_label = QLabel()
        self.img_label.setMinimumSize(1280, 720)
        self.img_label.setStyleSheet("""
            QLabel {
                border: none;
                border-radius: 20px;
                background-color: rgba(255,255,255, 0.4);
                padding: 4px;
            }
        """)
        self.img_label.setAlignment(Qt.AlignCenter)
        mid_layout.addWidget(self.img_label, stretch=1)

        # 右侧控制面板
        right_group = QGroupBox("⚙️ 系统控制")
        right_group.setFixedWidth(500)
        right_layout = QVBoxLayout(right_group)
        right_layout.setSpacing(5)

        right_layout.addWidget(QLabel("IOU 阈值"))
        self.iou_combo = QComboBox()
        self.iou_combo.addItems(["0.3","0.4","0.5","0.6","0.7"])
        self.iou_combo.setCurrentText("0.5")
        right_layout.addWidget(self.iou_combo)

        right_layout.addWidget(QLabel("YOLO 模型"))
        self.model_edit = QLineEdit("best.pt")
        right_layout.addWidget(self.model_edit)

        self.browse_btn = QPushButton("🧩 选择模型")
        self.img_btn = QPushButton("🖼️ 选择图片/视频")
        self.folder_btn = QPushButton("📂 选择文件夹")
        self.camera_btn = QPushButton("📷 打开摄像头")
        self.voice_btn = QPushButton("🔊 语音：开启")
        self.stop_btn = QPushButton("⏹️ 停止当前任务")
        self.save_btn = QPushButton("💾 导出记录")
        self.exit_btn = QPushButton("❌ 退出系统")

        right_layout.addWidget(self.browse_btn)
        right_layout.addWidget(self.img_btn)
        right_layout.addWidget(self.folder_btn)
        right_layout.addWidget(self.camera_btn)
        right_layout.addWidget(self.voice_btn)
        right_layout.addWidget(self.stop_btn)
        right_layout.addWidget(self.save_btn)
        right_layout.addStretch()
        right_layout.addWidget(self.exit_btn)

        mid_layout.addWidget(right_group)
        main_layout.addLayout(mid_layout)

        # 检测记录表格
        table_group = QGroupBox("📋 检测历史记录")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["序号","来源","识别结果","置信度","垃圾类别","坐标"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(280)
        table_layout.addWidget(self.table)
        main_layout.addWidget(table_group)

        # 按钮绑定
        self.browse_btn.clicked.connect(self.browse_model)
        self.img_btn.clicked.connect(self.open_media)
        self.folder_btn.clicked.connect(self.open_folder)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.voice_btn.clicked.connect(self.toggle_voice)
        self.stop_btn.clicked.connect(self.stop_all)
        self.save_btn.clicked.connect(self.save_records_to_csv)
        self.exit_btn.clicked.connect(self.close)

    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        self.voice_btn.setText("🔊 语音：开启" if self.voice_enabled else "🔇 语音：关闭")

    def load_model(self):
        try:
            if not self.model:
                self.model = YOLO(self.model_edit.text())
        except:
            QMessageBox.warning(self,"警告","模型加载失败")

    def browse_model(self):
        p, _ = QFileDialog.getOpenFileName(filter="*.pt")
        if p:
            self.model_edit.setText(p)
            self.model = None

    def open_media(self):
        p, _ = QFileDialog.getOpenFileName()
        if not p:return
        self.stop_all()
        self.reset_all()
        self.load_model()
        mode = "image" if p.lower().endswith(('png','jpg','jpeg')) else "video"
        self.thread = DetectorThread(mode, p, self.model, float(self.iou_combo.currentText()))
        self.thread.frame_ready.connect(self.show_result)
        self.thread.finished.connect(self.on_task_finished)
        self.thread.start()

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory()
        if not folder:return
        self.stop_all()
        self.reset_all()
        self.load_model()
        self.folder_thread = FolderProcessThread(folder, self.model, float(self.iou_combo.currentText()))
        self.folder_thread.progress.connect(self.show_folder_result)
        self.folder_thread.finished.connect(self.on_folder_finished)
        self.folder_thread.start()

    def toggle_camera(self):
        if not self.is_camera_running:
            self.stop_all()
            self.reset_all()
            self.is_camera_running = True
            self.camera_btn.setText("📷 关闭摄像头")
            self.load_model()
            self.thread = DetectorThread("camera", 0, self.model, float(self.iou_combo.currentText()))
            self.thread.frame_ready.connect(self.show_result)
            self.thread.start()
        else:
            self.stop_all()
            self.do_final_summary()

    def stop_all(self):
        if self.thread:
            self.thread.stop()
            self.thread = None
        if self.folder_thread:
            self.folder_thread.stop()
            self.folder_thread = None
        self.is_camera_running = False
        self.camera_btn.setText("📷 打开摄像头")

    def reset_all(self):
        self.all_detected_objects.clear()
        self.last_items.clear()
        self.ai_summary_edit.clear()
        self.table.setRowCount(0)

    def on_task_finished(self):
        self.do_final_summary()

    def on_folder_finished(self):
        self.do_final_summary()
        QMessageBox.information(self, "完成", "文件夹内所有图片检测完毕！")

    def do_final_summary(self):
        obj_list = list(self.all_detected_objects)
        if not obj_list:
            self.ai_summary_edit.setPlainText("未检测到任何垃圾目标")
            return

        # ===================== 核心优化：物品 + 分类 一一对应 =====================
        item_cat_lines = []
        for name in obj_list:
            cat = GARBAGE_TYPE_MAP.get(name, "未知类别")
            item_cat_lines.append(f"{name}（{cat}）")

        # 拼接成干净、清晰、不混乱的文本
        clean_summary_text = "、".join(item_cat_lines)
        prompt = f"{clean_summary_text}"

        # 发送给AI总结
        if self.ai_available:
            self.ai_thread = AISummaryThread(self.ai_client, prompt)
            self.ai_thread.reply_ready.connect(self.show_final_summary)
            self.ai_thread.start()
        else:
            self.show_final_summary(prompt)

    def show_final_summary(self, summary):
        self.ai_summary_edit.setPlainText(summary)
        if self.voice_enabled:
            self.voice_worker.add(summary)

    def show_folder_result(self, filename, qimg, dets, t):
        self.show_result(qimg, dets, t, filename)

    def show_result(self, qimg, res, t, source_name=None):
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.img_label.size(), Qt.KeepAspectRatio
        ))
        self.time_label.setText(f"⏱️ 推理用时：{t:.2f}s")

        if not res:
            self.num_label.setText("🎯 检测目标：0 个")
            return

        for r in res:
            self.all_detected_objects.add(r["name"])

        self.num_label.setText(f"🎯 检测目标：{len(res)} 个")

        r_first = res[0]
        name_first = r_first['name']
        cat_first = GARBAGE_TYPE_MAP.get(name_first, "未知")
        self.type_label.setText(f"👁️ 目标类型：{name_first}")
        self.conf_label.setText(f"📈 置信度：{r_first['conf']*100:.1f}%")
        self.category_label.setText(f"🗑️ 垃圾类别：{cat_first}")

        for r in res:
            name = r['name']
            key = (name, tuple(r['xyxy']))
            if key not in self.last_items:
                self.last_items.add(key)
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row,0,QTableWidgetItem(str(row+1)))

                if source_name:
                    path_str = source_name
                elif self.is_camera_running:
                    path_str = "摄像头"
                elif self.thread and hasattr(self.thread, 'source'):
                    path_str = os.path.basename(self.thread.source)
                else:
                    path_str = "文件"

                self.table.setItem(row,1,QTableWidgetItem(path_str))
                self.table.setItem(row,2,QTableWidgetItem(name))
                self.table.setItem(row,3,QTableWidgetItem(f"{r['conf']*100:.1f}%"))
                self.table.setItem(row,4,QTableWidgetItem(GARBAGE_TYPE_MAP.get(name,"未知")))
                self.table.setItem(row,5,QTableWidgetItem(str(r["xyxy"])))

    # ====================== 自动保存记录到 CSV ======================
    def save_records_to_csv(self):
        row_count = self.table.rowCount()
        if row_count == 0:
            QMessageBox.warning(self, "提示", "暂无检测记录可保存！")
            return

        save_dir = r"D:\DeepLearning\ultralytics-8.3.163\Garbage_Classification_Detection\my_system\save"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"垃圾分类检测记录_{timestamp}.csv")

        ai_summary = self.ai_summary_edit.toPlainText()

        try:
            with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["序号", "来源", "识别结果", "置信度", "垃圾类别", "坐标", "AI智能总结"])
                writer.writerow(["", "", "", "", "", "", ai_summary])
                writer.writerow([])

                for row in range(row_count):
                    items = []
                    for col in range(6):
                        item = self.table.item(row, col)
                        items.append(item.text() if item else "")
                    items.append("")
                    writer.writerow(items)

            QMessageBox.information(self, "保存成功", f"记录已自动保存至：\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"错误信息：{str(e)}")

    def closeEvent(self,e):
        self.stop_all()
        e.accept()

