# -*- coding: utf-8 -*-
# @Time : 2023/9/18 下午4:26
# @Author : Xiuxuan Shen

import matplotlib.pyplot as plt
from datetime import datetime

# 获取当前日期时间
current_datetime = datetime.now()

# 提取月份和日期
current_month = current_datetime.month
current_day = current_datetime.day

# 用于存储epoch和loss的列表
epochs = []
losses = []

# 打开文件并逐行读取
with open('/home/sxx/experiment/law/law_consistence/records/log/te/te_0304.log', 'r') as file:
    for line in file:
        # 解析每一行为字典
        log_entry = eval(line.strip())  # 使用eval来解析字典，假设文件中的每行都是有效的字典表示

        # 提取epoch和loss
        epoch = log_entry.get('epoch')
        loss = log_entry.get('train_loss')

        # 如果epoch和loss都存在，则添加到列表中
        if epoch is not None and loss is not None:
            epochs.append(epoch)
            losses.append(loss)

# 绘制loss曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='^', markersize=6, color='orange', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid(True)

plt.savefig('/home/sxx/experiment/law/law_consistence/records/curve/'+'train_loss_curve_'+str(current_month)+str(current_day)+'.png')
plt.show()
