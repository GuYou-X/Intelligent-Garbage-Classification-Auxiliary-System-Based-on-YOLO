# Intelligent Garbage Classification Auxiliary System Based on YOLO
## 基于YOLO的智能垃圾分类辅助系统
本科毕业设计 | 多版本YOLO对比训练 + PyQt5桌面可视化一体化项目

## 项目整体目录总览
项目根目录：`Garbage_Classification_Detection`
分为两大模块：
1. `my_system`：PyQt5可视化GUI推理系统（主运行程序）
2. `my_train`：数据集处理、模型训练、指标分析全套实验代码

### 1. my_system 目录结构（GUI推理端）
```
Garbage_Classification_Detection/my_system/
├── __pycache__/                # Python缓存文件
├── save/                       # 系统导出的检测记录CSV、截图保存目录
├── test/                       # 测试素材文件夹（图片/视频）
├── background.png              # UI界面背景图
├── best.pt                     # 默认最优模型权重(yolo12s)
├── best-yolo11n.pt / best-yolo11s.pt
├── best-yolo12n.pt / best-yolo12s.pt
├── best-yolo5nu.pt / best-yolo5su.pt
├── best-yolo8n.pt / best-yolo8s.pt
├── best-yolo10n.pt / best-yolo10s.pt  # 全部训练完成的各版本轻量化权重
├── config.py                   # 全局配置文件（路径、类别、LLM参数）
├── detector.py                 # YOLO推理引擎封装
├── main.py                     # GUI程序入口，直接运行启动软件
└── ui.py                       # PyQt5界面渲染、信号槽逻辑
```

### 2. my_train 目录结构（数据集&训练实验端）
```
Garbage_Classification_Detection/my_train/
├── balanced_garbage_datasets/        # 均衡处理后标准训练验证数据集
├── balanced_garbage_datasets_analysis_plots/ # 均衡后类别分布统计图
├── cfg/                              # YOLO训练yaml配置文件
├── cleaned_garbage_datasets/         # 清洗去重后的原始数据集
├── cleaned_garbage_datasets_analysis_plots/  # 清洗后分布统计图
├── garbage_datasets/                 # 原始开源垃圾数据集
├── garbage_datasets_analysis_plots/  # 原始数据集分布统计图
├── predict/                          # 单张图片预测测试输出文件夹
├── results/                          # 全部模型训练日志、损失曲线、指标图表
├── val/                              # 模型性能评估输出，和results内容重合
├── cam.py                            # 摄像头检测脚本
├── classes_count_balanced.py         # 统计均衡数据集各类样本数量
├── classes_count_cleaned.py          # 统计清洗数据集各类样本数量
├── datasets_balance.py                # 数据集分层均衡重划分脚本
├── datasets_clean.py                 # 数据集清洗、删除损坏图像/冗余标注
├── datasets_verify.py                 # 图像&标注完整性校验脚本
├── predict.py                        # 单模型批量预测测试脚本
├── train.py                          # 多版本YOLO统一训练入口脚本
├── val.py                            # 性能评估脚本
├── yolo11m.pt / yolo11n.pt / yolo11s.pt
├── yolo12m.pt / yolo12n.pt / yolo12s.pt
├── yolov5m6u.pt / yolov5mu.pt / yolov5n6u.pt / yolov5nu.pt / yolov5s6u.pt / yolov5su.pt
├── yolov8m.pt / yolov8n.pt / yolov8s.pt / yolov10m.pt / yolov10n.pt / yolov10s.pt  # 完整训练权重文件
```

## 项目概述
伴随国内城市化持续推进，生活垃圾清运量逐年上涨，传统人工垃圾分类存在人力成本高、识别误差大、普及困难、效率低下等痛点。本课题结合深度学习单阶段目标检测技术，完整实现**数据集处理-多模型对比训练-桌面端智能辅助系统**全流程毕业设计方案。

完整研究开发链路：
1. 数据集自动化校验、清洗、分层均衡重划分，生成标准化训练数据集；
2. 自建实拍独立测试集，用于客观评测模型泛化能力；
3. 统一超参数批量训练 YOLOv5 / YOLOv8 / YOLOv10 / YOLOv11 / YOLO12 多规格轻量化模型，横向对比精度、召回、mAP、推理速度；
4. 筛选综合性能最优 YOLO12s 作为核心推理模型；
5. 基于PyQt5开发多线程GUI可视化系统，解决推理卡顿问题；
6. 集成图片/视频/摄像头实时检测、LLM智能分类解读、TTS语音播报、检测记录本地CSV导出完整功能；
7. 多场景综合性能测试，验证系统识别精度、实时性、长期运行稳定性。

系统可识别40类生活细分类垃圾，自动归类为**可回收垃圾、有害垃圾、厨余垃圾、其他垃圾**四大国标类别，结合大语言模型文本总结与语音播报，为普通居民提供轻量化、易上手的垃圾分类智能辅助工具。

## 环境依赖
### 硬件实验环境
- GPU：NVIDIA RTX 4070 Laptop 8G
- 内存：16GB DDR5
- 存储：2TB SSD

### 软件版本
```
Python 3.11.13
PyTorch 2.5.0 + CUDA 11.8
Ultralytics 8.3.163
PyQt5
OpenCV-Python
NumPy / Matplotlib / Pandas
Windows内置TTS语音库
```

### 一键安装依赖
```bash
# 安装GPU版Pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 其余深度学习、GUI、数据处理依赖
pip install ultralytics pyqt5 opencv-python numpy matplotlib pandas
```

## 数据集介绍
### 1. 开源基础数据集
数据源：https://ai.gitcode.com/ai53_19/garbage_datasets
- 原始训练集：16840张，验证集：1776张
- 总计40细分类垃圾，分为四大国标类别：
  - 可回收垃圾(23类)：充电宝、塑料瓶、纸箱、金属罐、旧衣物等
  - 有害垃圾(3类)：干电池、过期药品、药膏
  - 厨余垃圾(8类)：果皮、骨头、剩饭、菜叶、蛋壳、鱼骨等
  - 其他垃圾(6类)：快餐盒、烟头、牙签、污损塑料、花盆、竹筷
- 标准YOLO txt归一化标注格式，可直接用于模型训练

### 2. 数据集预处理完整流程（my_train内脚本一键执行）
1. `datasets_verify.py`：自动化校验，过滤损坏图像、无匹配标签冗余txt文件；
2. `datasets_clean.py`：清洗冗余文件，生成干净无报错数据集；
3. `classes_count_cleaned.py`：统计清洗后各类样本数量，生成分布统计图；
4. `datasets_balance.py`：按8:2分层随机重划分训练/验证集，均衡各类样本数量，缓解长尾分布；
5. `classes_count_balanced.py`：输出均衡后数据集样本分布图表。

### 3. 自建独立测试集
手机实拍居家、厨房、楼道、小区垃圾桶、户外堆放等真实场景；覆盖逆光、暗光、多垃圾堆叠、杂物遮挡、小目标密集等复杂工况；**全程不参与训练**，仅用于客观评测各模型泛化能力，规避过拟合带来的评测偏差。

## 模型训练实验（my_train）
### 参与对比模型规格
YOLOv5(nu/su/m6u) / YOLOv8(n/s/m) / YOLOv10(n/s/m) / YOLO11(n/s/m) / YOLO12(n/s/m)

### 统一训练超参数（控制变量保证对比公平）
```
epochs=100, patience=15
batch=16, imgsz=640
optimizer=AdamW, lr0=0.01, lrf=0.01
box=7.5, cls=0.5, dfl=1.5
iou=0.7, pretrained=True, amp=True
device=0, workers=0
```

### 实验结论
1. 同系列s标准版模型精度全面优于n轻量化版本；
2. YOLO迭代版本综合性能稳步提升，**YOLO12s** 在精确率、召回率、mAP@0.5、mAP@0.5-0.95四项指标全部最优；
3. 所有n/s轻量化模型单帧推理耗时约40ms，实时帧率≥30FPS，满足PC端实时检测需求；
4. GUI系统默认加载 YOLO12s 作为核心推理权重。

## GUI系统核心功能（my_system）
### 1. 多模型与推理参数配置
- 支持YOLOv5/v8/v10/v11/v12全系列权重文件自由切换加载；
- 可视化滑动条自定义IOU阈值，优化重叠、遮挡垃圾的漏检/误检问题。

### 2. 多源媒体检测输入
- 单张图片、单个视频文件离线检测；
- 文件夹批量图片批量推理；
- 本地摄像头实时视频流动态识别。

### 3. 检测结果可视化展示
- 画面实时绘制检测框、类别标签、置信度数值；
- 界面实时展示推理耗时、目标总数、各类别统计信息；
- 底部表格自动缓存全部检测历史记录，全程可追溯。

### 4. LLM智能总结模块
检测任务结束后自动汇总所有识别垃圾信息，拼接标准化提示词调用本地大模型生成垃圾分类科普与投放指引；内置降级方案，大模型服务异常时自动切换规则文本输出，保证功能可用。

### 5. 实时语音反馈
基于Windows原生TTS语音合成，独立消息队列线程异步播报，不阻塞检测推理流程；可手动开关语音，播报单目标识别结果与整体AI分类总结。

### 6. 检测记录导出存储
自动记录序号、数据来源、垃圾类别、置信度、目标坐标、LLM总结全文；一键导出带时间戳CSV文件，Excel可直接打开，方便数据归档与统计分析，文件默认保存至`my_system/save/`。

## 运行操作指南
### 一、数据集预处理（进入my_train文件夹执行）
```bash
# 1. 校验原始数据集图像与标注匹配性
python datasets_verify.py
# 2. 清洗损坏文件、冗余标注
python datasets_clean.py
# 3. 分层均衡重划分训练/验证集
python datasets_balance.py
```

### 二、批量训练所有YOLO模型（my_train）
```bash
python train.py
```
按照统一超参数完成多模型训练，训练日志、损失曲线、指标图表自动保存至`my_train/results`文件夹。

### 三、启动可视化GUI系统（进入my_system文件夹执行）
```bash
python main.py
```
软件操作流程：
1. 加载训练完成的YOLO权重文件；
2. 拖动滑块调节IOU阈值适配遮挡/密集场景；
3. 选择图片/视频/摄像头三种检测模式开始推理；
4. 实时查看带标注可视化画面、AI分类总结、语音播报；
5. 一键导出全部历史检测记录至save文件夹。

## 系统综合测试效果
1. 推理性能：轻量化模型单帧推理40ms，摄像头实时帧率稳定30FPS以上，多线程架构界面无卡顿；
2. 识别精度：YOLO12s在自建复杂场景测试集综合mAP＞0.67，对遮挡、小目标、相似垃圾识别效果最优；
3. 稳定性：支持长时间连续摄像头检测，无内存泄漏、程序闪退、线程残留问题；
4. 易用性：界面分区清晰，操作按钮直观，无计算机基础普通用户可快速上手。

## 现存不足与后续改进方向
### 当前存在缺陷
1. 数据集小众冷门垃圾样本偏少，类别长尾分布未完全消除，少量类别易出现漏检；
2. 由于时间、硬件成本，本课题仅对比n/s轻量化模型，未深度挖掘m中大型网络的精度上限；
3. 系统仅适配Windows PC本地运行，未做模型量化、剪枝，无移动端、嵌入式边缘设备部署方案。

### 后续优化展望
1. 扩充多场景实拍垃圾分类数据集，定向补充冷门类别样本，引入半监督学习降低人工标注成本；
2. 对YOLO网络进行定制改进，嵌入轻量化注意力机制、小目标增强模块，进一步提升遮挡、易混淆垃圾识别精度；
3. 模型量化、剪枝压缩，适配树莓派、Android移动端等边缘终端，开发轻量化离线检测程序；
4. 搭建云边协同智慧环卫架构，边缘端本地实时检测，云端汇总垃圾分类大数据、在线迭代更新模型。

## .gitignore 仓库忽略配置（直接复制到根目录.gitignore）
```gitignore
my_system/__pycache__/
my_system/save/
my_system/test/images1/
my_system/test/images2/
my_system/test/videos/

my_train/balanced_garbage_datasets
my_train/balanced_garbage_datasets_analysis_plots
my_train/cleaned_garbage_datasets
my_train/cleaned_garbage_datasets_analysis_plots
my_train/garbage_datasets
my_train/garbage_datasets_analysis_plots
my_train/predict
my_train/results
my_train/val

*.pt
```

## 开源说明
本项目为个人毕业设计开源项目，仅用于学术学习、技术交流参考，禁止抄袭。

## 致谢
感谢毕业设计指导老师在课题选题、数据集构建、多模型对比实验、GUI系统开发全过程给予的专业指导与帮助。
