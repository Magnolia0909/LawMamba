# -*- coding: utf-8 -*-

import csv
import json

# 读取CSV文件并保存为JSON
def csv_to_json(csv_filename, json_filename):
    data_list = []

    with open(csv_filename, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        i = 0
        for row in csv_reader:
            print(row)
            if len(row) >= 3:
                data_dict = {
                    "ids": int(i/3),
                    "contract": row[0],
                    "law_content": row[1],
                    "law_id": int(row[2]),
                    "law_label": int(row[3]),
                    "law_num": int(row[4]),
                    "rank": int(row[5]),
                }
                data_list.append(data_dict)
                i = i + 1

    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

# 指定CSV和JSON文件名
'''
  测试集
'''
# csv_filename = '/home/sxx/experiment/data/law_data/te/test.csv'
# json_filename = '/home/sxx/experiment/data/law_data/te/test.json'

'''
  训练集
'''
csv_filename = '/home/sxx/experiment/data/law_data/te/mydata.csv'
json_filename = '/home/sxx/experiment/data/law_data/te/train.json'

# 调用函数进行转换
csv_to_json(csv_filename, json_filename)

print("转换完成！")
