# -*- coding: utf-8 -*-
# @Time : 2023/9/1 下午1:03
# @Author : Xiuxuan Shen
import json


# 读取JSON文件并处理数据
def process_json(json_filename, output_filename):
    with open(json_filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    processed_data = []

    for entry in data:
        label_value = entry.get("label")

        # 判断 "label" 是否是整数
        if label_value is not None and label_value.isdigit():
            entry["label"] = int(label_value)  # 将字符串转换为整数
            processed_data.append(entry)

    with open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(processed_data, output_file, ensure_ascii=False, indent=4)


# 指定JSON文件名和输出文件名
json_filename = '../data/te/train.json'
output_filename = '../data/te/train.json'

# 调用函数进行数据处理
process_json(json_filename, output_filename)

print("数据处理完成！")
