import os
import random

import cv2 as cv
import numpy as np
import pandas as pd


def data_from_excel(excel_name):
    selected_columns = ['Time', 'FHR']
    data = pd.read_csv(excel_name, usecols=selected_columns)
    return data


def data_to_complex_pic(width, height, grid_line_color, random_spot_color, img_complex_name, background_img):
    # 创建了一个具有指定高度（height）、宽度（width）和深度（3，通常用于表示RGB颜色值）的三维NumPy数组，数组中所有元素都初始化为零
    # dtype=np.uint8 指定数组的数据类型为无符号8位整数，通常用于表示像素值在0到255之间的图像数据

    # 在x轴上每60秒绘制纵向垂直线
    for x in np.arange(60, 1200, 60):
        cv.line(background_img, (x, 0), (x, height), color=grid_line_color, thickness=1)
    # 线宽为1个像素
    # (x, 0)垂直线最下面的点, (x, height)垂直线最上面的点

    # 在y轴上每10bpm绘制横向线
    for y in np.arange(10, 200, 10):
        cv.line(background_img, (0, y), (width, y), color=grid_line_color, thickness=1)

    # y_list = [int(height - i) for i in data['FHR']]
    # 计算了data['FHR']中的每个元素i与height的差值，并将结果强制转换为整数
    # 这个列表中的每个元素都代表了根据'FHR'数据计算出的新的纵坐标值

    # 图像 background_img 上的坐标 (x, y) 处设置了一个？色（(R, G, B)）的像素值

    for _ in range(100):
        background_img[np.random.randint(0, height)][np.random.randint(0, width)] = random_spot_color
    # 循环迭代100次，每次迭代中会随机选择图像 background_img 中的一个像素，并将该像素的颜色设置为？色 (255, 255, 255)
    # 使用 np.random.randint(0, height) 来随机选择一个高度坐标，然后使用 np.random.randint(0, width) 来随机选择一个宽度坐标
    # 将该坐标处的像素颜色设置为？色

    cv.imwrite(img_complex_name, background_img)
    img_complex_array = cv.imread(img_complex_name)
    return img_complex_array


def data_to_normal_pic(img_normal_name, background_img):
    cv.imwrite(img_normal_name, background_img)
    img_normal_array = cv.imread(img_normal_name)
    return img_normal_array


def get_background_with_pts(background_color, curve_line_color, data, height, width):
    background_img = np.zeros((height, width, 3), dtype=np.uint8)
    background_img[:, :] = background_color
    # y_list = [int(height - i) for i in data['FHR']]
    y_list = []
    for i in data['FHR']:
        if i > 0:
            y_list.append(int(height - i))
        else:
            y_list.append(0)
    for x, y in enumerate(y_list):
        background_img[y][x] = curve_line_color
    return background_img


def random_rgb_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


# 使用 random.randint(0, 255) 为R、G、B通道分别生成一个0到255之间的随机整数，将这三个值组合成一个元组作为RGB颜色返回。每次调用这个函数都会生成一个新的随机RGB颜色

if __name__ == '__main__':

    folder_path = './sub_excel'
    a_pic_path = './a_pic_out'
    b_pic_path = './b_pic_out'
    # pic_path = './pic_out'

    # 黄色(255, 255, 0)
    # 蓝色(0, 0, 255)
    # 白色(255, 255, 255)
    # 粉红色(220,20,60)
    background_color = (255, 255, 255)
    curve_line_color = (0, 0, 255)
    grid_line_color = (220, 20, 60)
    # random_spot_color = (0, 0, 0)
    random_spot_color = random_rgb_color()

    width, height = 1200, 200

    # 创建字典来存储img_complex，以img_complex_name作为键
    img_complex_dict = {}
    img_normal_dict = {}

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            excel_name = os.path.join(folder_path, file)
            data = data_from_excel(excel_name)
            img_complex_name = os.path.join(a_pic_path, file[:-4] + '.png')
            img_normal_name = os.path.join(b_pic_path, file[:-4] + '.png')
            background_img = get_background_with_pts(background_color, curve_line_color, data, height, width)

            img_complex_array = data_to_complex_pic(width, height, grid_line_color, random_spot_color, img_complex_name,
                                                    background_img.copy())
            img_normal_array = data_to_normal_pic(img_normal_name, background_img.copy())

            # 将img_complex_array存储到字典中以img_complex_name为键
            img_complex_dict[img_complex_name] = img_complex_array
            img_normal_dict[img_normal_name] = img_normal_array

cv.waitKey(0)
cv.destroyAllWindows()
