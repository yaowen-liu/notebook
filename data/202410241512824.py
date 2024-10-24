from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
from models import femnist_model,femnist_model_CNN
import flip_prune
from models import LaplaceFast
from calculate_size import get_model_size2,convert_bits_to_human_readable
#coding=utf-8
import os,pickle, gc, time,csv
import math
from tkinter import NO
import data_utils
from sklearn.manifold import TSNE
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization,concatenate
from tensorflow.keras.models import Model
import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, Callable, Optional, Tuple, List
from flwr.common import (
    # EvaluateIns,
    # EvaluateRes,
    # FitIns,
    # FitRes,
    # MetricsAggregationFn,
    # Parameters,
    # Scalar,
    # Weights,
    parameters_to_weights,
    weights_to_parameters,
)

"""这个是对CNN的可视化"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tempfile
import os
from tensorflow.keras.models import load_model
data_dir  = r'D:\Work_kong\FL_prunning\fl_sever_version\fuben\Ablation_EX_changewave_15DATA\Double_wave\symX2系列\sym8_sym12_57x57_15'
data_dir2  = r'D:\Work_kong\FL_prunning\fl_sever_version\fuben\WTG_32_10_10'


CNN_aggregated_weights = np.load(r"D:\Work_kong\FL_prunning\fl_sever_version\fuben\WTG_CNN_feature_text.npy", allow_pickle=True)

client_aggregated_weights=np.load(r"D:\Work_kong\FL_prunning\fl_sever_version\fuben\客户端剪枝模型的保存\WTG_client_model_weights_0.9.npy", allow_pickle=True)
# print(LCNN_aggregated_weights)
model, out_model = femnist_model_CNN()
model.set_weights(CNN_aggregated_weights)
out_model.set_weights(CNN_aggregated_weights)
# """测试客户端实际剪枝后模型的大小"""
# # keras_file = r'D:\models\WTG_custom_model_0.9.h5'
# # tf.keras.models.save_model(model, keras_file, include_optimizer=False)
# # print('Saved baseline model to:', keras_file)
#
# def get_gzipped_model_size(file):
#   import os
#   import zipfile
#
#   _, zipped_file = tempfile.mkstemp('.zip')
#   with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
#     f.write(file)
#
#   return os.path.getsize(zipped_file)
#
# dirrr = r"D:\models\WTG_custom_model_0.9.h5"
# print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(dirrr)))





# DIR = r"D:\models\client_model.h5"
# # model.save(DIR)
# model_file_size = os.path.getsize(DIR)
# print("Model file size:", model_file_size, "bytes")
# import sys
# memory = sys.getsizeof(DIR)
# print("Model memory size:", memory, "bytes")
"""混淆矩阵neg"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
labels = ['H', 'PF', 'CF', 'RB', 'WF', 'BW']
x_val, y_val = data_utils.get_all_test_data(data_dir2)
y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
# 计算混淆矩阵
conf_matrix = confusion_matrix(y_val, y_pred_classes)
# print("Confusion Matrix:")
# print(conf_matrix)

plt.figure(figsize=(12, 9))
heatmap = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 28, "fontname": "Times New Roman"}, vmin=0, vmax=200,  cbar_kws={"ticks": range(0, 201, 40)})
plt.xlabel('Predicted Label', fontdict={'fontname': 'Times New Roman', 'fontsize': 32})
plt.ylabel('True Label', fontdict={'fontname': 'Times New Roman', 'fontsize': 32})

# 自定义标签
plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, fontname='Times New Roman', fontsize=30)
plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, fontname='Times New Roman', fontsize=30)

# 获取颜色条对象
cbar = heatmap.collections[0].colorbar

# 设置刻度位置和标签
# 设置刻度位置和标签

cbar.ax.tick_params(labelsize=26)
cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontname="Times New Roman")
plt.savefig('WTG_FedAvg_confusion_matrix.svg', format = 'svg')

plt.show()

# 使用T-SNE进行降维
# 使用T-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(y_pred)  # 这里使用了模型的预测结果，你也可以根据需要使用其他数据
plt.rcParams['font.family'] = 'Times New Roman'
# 创建散点图
plt.figure(figsize=(10, 8))

# 为每个类别添加散点和标签
for i, label in enumerate(labels):
    # 获取当前类别的样本索引
    idx = y_val == i

    # 绘制散点图，并使用不同颜色表示不同类别
    plt.scatter(x_tsne[idx, 0], x_tsne[idx, 1], label=label, cmap='viridis')

    # 在每个类别的中心位置添加文本标签
    x_center = x_tsne[idx, 0].mean()
    y_center = x_tsne[idx, 1].mean()
    plt.text(x_center, y_center, label, fontsize=22, fontname='Times New Roman', ha='center', va='center', color='black')

# 自定义边界
custom_x_min, custom_x_max = -70, 70
custom_y_min, custom_y_max = -70, 70

# 自定义刻度
x_ticks = [-60, -30, 0, 30, 60]
y_ticks = [-60, -30, 0, 30, 60]

# 设置 X 轴和 Y 轴刻度
plt.xticks(x_ticks, fontsize=26)
plt.yticks(y_ticks, fontsize=26)

# 设置图的边界
plt.xlim(custom_x_min, custom_x_max)
plt.ylim(custom_y_min, custom_y_max)

plt.xlabel(' 1st dimension', fontdict={'fontname': 'Times New Roman', 'fontsize': 32})
plt.ylabel(' 2nd dimension', fontdict={'fontname': 'Times New Roman', 'fontsize': 32})

plt.legend(fontsize=18)
plt.savefig('WTG_FedAvg_T_SENT.svg', format = 'svg')
plt.show()