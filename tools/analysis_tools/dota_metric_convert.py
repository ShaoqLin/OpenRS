# 处理同一个数据集下多个json文件时，仅运行一次class_txt即可
import json
import os


"存储标签与预测框到txt文件中"
def json_txt(json_path, txt_path):
    "json_path: 需要处理的json文件的路径"
    "txt_path: 将json文件处理后txt文件存放的文件夹名"
    # 生成存放json文件的路径
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    # 读取json文件
    with open(json_path, 'r') as f:
        dict = json.load(f)
    # 得到images和annotations信息

    # 使用images下的图像名的id创建txt文件
    list=[]  # 将文件名存储在list中
    for i in range(len(dict)):
        image_id = dict[i].get("image_id")
        score = dict[i].get("score")
        bbox = dict[i].get("bbox")
        category_id = dict[i].get("category_id")
        open(txt_path + str(category_id.get("id")) + '.txt', 'w')
        list.append(i.get("id"))


    # 将id对应图片的bbox写入txt文件中
    for i in list:
        for j in annotations_value:
            if j.get("image_id") == i:
                # bbox标签归一化处理
                num = sum(j.get('bbox'))
                new_list = [round(m / num, 6) for m in j.get('bbox')]  # 保留六位小数
                with open(txt_path + str(i) + '.txt', 'a') as file1:  # 写入txt文件中
                    print(j.get("category_id"), new_list[0], new_list[1], new_list[2], new_list[3], file=file1)


"将id对应的标签存储在class.txt中"
def class_txt(json_path, class_txt_path):
    "json_path: 需要处理的json文件的路径"
    "txt_path: 将json文件处理后存放所需的txt文件名"
    # 生成存放json文件的路径
    with open(json_path, 'r') as f:
        dict = json.load(f)
    # 得到categories下对应的信息
    categories_value = dict.get("categories")  # 得到某个键下对应的值
    # 将每个类别id与类别写入txt文件中
    with open(class_txt_path, 'a') as file0:
        for i in categories_value:
            print(i.get("id"), i.get('name'), file=file0)


json_txt("work_dirs/benchmark_faster-rcnn_1gpu_bs4_r50_fpn_1x_lr1e-3_dotav1_0/output.bbox.json",
         "work_dirs/benchmark_faster-rcnn_1gpu_bs4_r50_fpn_1x_lr1e-3_dotav1_0/")
# class_txt("eval.json", "id_categories.txt")
