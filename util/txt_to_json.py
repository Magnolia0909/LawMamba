import json
def txt_to_json(txt_filename, json_filename):
    data_list = []

    with open(txt_filename, 'r', encoding='utf-8') as txt_file:
        for line in txt_file:
            line = line.strip()  # 去除首尾空白字符
            elements = line.split()  # 使用空格分割元素

            # 将元素列表转换为包含指定键值的字典
            data_dict = {
                "premise": elements[0],
                "hypothesis": elements[1],
                "label": elements[2]
            }
            data_list.append(data_dict)

    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

txt_filename = '/home/sxx/experiment/data/law_data/CSTS/Chinese-STS-B/sts-b-test.txt'
json_filename = '/home/sxx/experiment/data/law_data/te/test.json'

txt_to_json(txt_filename, json_filename)

print("转换完成！")

