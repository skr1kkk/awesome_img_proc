import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime



def data_from_excel(excel_name):
    selected_columns = ['Time', 'FHR']
    data = pd.read_csv(excel_name, usecols=selected_columns)

    # time_strings = data['Time']
    # # 将时间字符串解析为datetime对象
    # time_objects = [datetime.strptime(time_str, "%Y/%m/%d %H:%M") for time_str in time_strings]
    #
    # data['time_in_seconds'] = (data['time'] - data['time'].min()).dt.total_seconds()
    # # # 将datetime对象转换为数值（秒数）
    # # time_values = [(time - time_objects[0]).total_seconds() for time in time_objects]
    # data['Time'] = data['time_in_seconds']

    return data


def data_to_complex_pic(df, dpi, background_color, line_color, pic_name):

    # plt.gca().set_facecolor(background_color)  # 设置背景颜色
    # plt.gca().set_facecolor('lightgray')  # 设置背景颜色，比如浅灰色

    # plt.xlabel('time(s)')  # 设置x轴标签
    # plt.ylabel('FHR(bpm)')  # 设置y轴标签
    # plt.title('Plot of Column1 vs Column2')  # 设置标题

    # # 添加其他点
    # plt.scatter(x_points, y_points, color='red')  # 可以设置点的颜色
    width, height = 1200, 200
    background_img = np.zeros((height, width, 3), dtype=np.uint8)

    # cv.circle(background_img, (x, y), radius=1, color=(0,0,255), thickness=-1)

    # inches_width = pixel_width / dpi
    # inches_height = pixel_height / dpi
    # fig, ax = plt.subplots(figsize=(inches_width, inches_height), dpi=100)  # 设置图形大小和分辨率
    #
    # plt.xlim(0, 1199)  # 设置x轴范围
    # plt.ylim(0, 199)  # 设置y轴范围
    # plt.axis('off')  # 不显示坐标轴

    # plt.figure(figsize=, dpi=dpi)  # 设置图像大小

    # 隐藏坐标轴和其他图示元素
    # ax.axis('off')
    # plt.figure(facecolor=background_color)

    # 在x轴上每60秒绘制纵向线
    for x in np.arange(60, 1200, 60):
        # plt.axvline(x, color=(0, 0, 0))
        cv.line(background_img, (x, 0), (x, height), color=(255,255,0), thickness=1)
    # 在y轴上每10bpm绘制横向线
    for y in np.arange(10, 200, 10):
        # plt.axhline(y, color=(0, 0, 0))
        cv.line(background_img, (0, y), (width, y), color=(255,255,0), thickness=1)
    y_list = [int(height - i) for i in df['FHR']]
    for x, y in enumerate(y_list):
        background_img[y][x] = (0, 0, 255)
    for _ in range(100):
        background_img[np.random.randint(0, height)][np.random.randint(0, width)] = (255,255,255)

    cv.imwrite(pic_name, background_img)
    img = cv.imread(pic_name)
    #
    # # plt.figure(figsize=(inches_width, inches_height), dpi=dpi)  # 设置图像大小

    # 将字符串时间数据转换为数值时间数据
    time_data = [i for i in range(1, 1200)]

    # x_list = [i for i in range(len(df['FHR']))]
    # plt.plot(x_list, df['FHR'], color=line_color)  # 绘制曲线
    # plt.plot(df['Time'], df['FHR'], color=line_color)  # 绘制曲线


    # # # 设置格子线：本身并不改变刻度的位置，只是在现有的刻度位置上添加参考线
    # # # plt.grid(True) #添加默认样式的格子线
    # # plt.grid(True, color=(0, 0, 0))
    # # # 设置x轴和y轴的格子间隔：调整刻度的位置，进而间接影响格子线的布局
    # # plt.xticks(np.arange(0, 1201, 60))  # 假设每分钟有60秒，设置x轴格子线间隔为每分钟
    # # plt.yticks(np.arange(0, 201, 20))  # 设置y轴格子线间隔为20bpm
    #

    # plt.show()
    # plt.savefig(pic_name, bbox_inches = 'tight', pad_inches = 0, dpi = 100)


if __name__ == '__main__':

    folder_path = './sub_excel'
    pic_path = './pic_out'

    background_color = (255, 255, 255)
    # 将其转换为0到1范围内的值
    background_color_normalized = tuple(c / 255 for c in background_color)
    line_color = (255, 0, 0)
    line_color_normalized = tuple(c / 255 for c in line_color)

    pixel_width = 1200
    pixel_height = 200
    dpi = 100

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            excel_name = os.path.join(folder_path, file)
            data = data_from_excel(excel_name)
            pic_name = os.path.join(pic_path, file[:-4] + '.png')
            data_to_complex_pic(data, dpi, background_color_normalized, line_color_normalized, pic_name)

cv.waitKey(0)
cv.destroyAllWindows()


#
# import os
#
# import
# import cv2 as cv
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from
#
# is_need_save = True  # 快捷保存，需要时就=True
# # 函数get_save_path()：保存路径设定为在本py文件同级下的imgs_scan文件夹
# imgs_root = './pic_out'  # 存储图片的子目录：本py文件同级下的pic_scan文件夹
#
#
# def get_save_path(img_name):
#     return imgs_root + '/' + img_name
#
#
# def data_from_excel(excel_path, col1, col2):
#     """
#     从Excel文件中读取两列数据并绘图。
#
#     :param excel_path: Excel文件的路径。
#     :param col1: 第一列的列名。
#     :param col2: 第二列的列名。
#     """
#     # 读取Excel文件
#     # df = pd.read_excel(excel_path)
#     df = pd.read_csv(excel_path)
#
#     plt.close()
#
#
# # 设置Excel文件所在的文件夹路径
# folder_path = 'path/to/excel/folder'  # 替换为实际路径
#
# # 遍历文件夹中的所有Excel文件
# for file in os.listdir(folder_path):
#     if file.endswith('.xlsx'):
#         plot_from_excel(os.path.join(folder_path, file), 'Column1', 'Column2')  # 替换'Column1'和'Column2'为实际列名
#
#     """
#     确保替换'path/to/excel/folder', 'Column1', 和'Column2'为实际的文件夹路径和列名。
#     这段代码会为文件夹中的每个Excel文件生成一个图表，并将图表保存在同一文件夹中，文件名与Excel文件相同，但扩展名为.png。
#     请注意，这个代码示例假定所有Excel文件的结构都是相同的，即它们都包含您要绘制的两列数据。如果文件结构不同，您可能需要对代码进行适当调整。
#     """
#
#     # 获取不含扩展名的文件名
#     file_name_without_ext = os.path.splitext(os.path.basename(excel_path))[0]
#
#     # 保存图表，文件名与Excel文件相同
#     plt.savefig(os.path.join(os.path.dirname(excel_path), file_name_without_ext + '.png'))
#     plt.close()
#
# # 设置Excel文件所在的文件夹路径
# folder_path = 'path/to/excel/folder'  # 替换为实际路径
#
# # 遍历文件夹中的所有Excel文件
# for file in os.listdir(folder_path):
#     if file.endswith('.xlsx'):
#         plot_from_excel(os.path.join(folder_path, file), 'Column1', 'Column2')  # 替换'Column1'和'Column2'为实际列名
#
#
# def create_plot(pixel_width, pixel_height, dpi=100):
#     """
#     创建一个具有特定像素尺寸的图像。
#
#     :param pixel_width: 图像的宽度（像素）
#     :param pixel_height: 图像的高度（像素）
#     :param dpi: 图像的分辨率（每英寸点数）
#     """
#     # 将像素尺寸转换为英寸
#     inches_width = pixel_width / dpi
#     inches_height = pixel_height / dpi
#
#     # 创建图形
#     fig = plt.figure(figsize=(inches_width, inches_height), dpi=dpi)
#
#     return fig
#     # 在这里绘制你的图像内容
#     # 例如：plt.plot([0, 1, 2], [0, 1, 0])
#     # # 显示图像
#     # plt.show()
#
#
# # # 创建一个800x600像素的图像
# # create_plot(800, 600)
#
#
# def plot_data(t, f, background_color=(1, 1, 1), line_color=(0, 0, 0)):
#     """
#     绘制数据图像。
#
#     :param t: 时间长度（秒），也是数据点的数量
#     :param f: 数据点的最大值
#     :param background_color: 背景颜色，格式为(R, G, B)
#     :param line_color: 线条颜色，格式为(R, G, B)
#     """
#     # 生成模拟数据
#     data = np.random.randint(0, f, t)
#
#     # 创建图像
#     plt.figure(figsize=(t / 100, f / 100))  # 图像大小按照像素数来设置
#     plt.plot(data, color=line_color)  # 绘制曲线
#     plt.xlim(0, t)  # 设置x轴范围
#     plt.ylim(0, f)  # 设置y轴范围
#     plt.axis('off')  # 不显示坐标轴
#     plt.gca().set_facecolor(background_color)  # 设置背景颜色
#
#     plt.xlabel("time(/s)")
#     plt.ylabel("FHR(/bpm)")
#     plt.title('FHR Result campare')
#
#     # 绘制图表
#     plt.figure()
#     plt.plot(df[col1], df[col2])
#
#     # 保存图表
#     plt.savefig(pic_path.replace('.xlsx', '.png'))
#     # 显示图像
#     plt.show()
#
#     # # 示例：绘制一个10秒长、数据范围为100的图像，白色背景和黑色线条
#     # plot_data(10, 100, background_color=(1, 1, 1), line_color=(0, 0, 0))
#
#     # 设置格子线：本身并不改变刻度的位置，只是在现有的刻度位置上添加参考线
#     # plt.grid(True) #添加默认样式的格子线
#     plt.grid(True, color=(1, 0, 0))
#
#     # 设置x轴和y轴的格子间隔：调整刻度的位置，进而间接影响格子线的布局
#     plt.xticks(np.arange(0, t + 1, 60))  # 假设每分钟有60秒，设置x轴格子线间隔为每分钟
#     plt.yticks(np.arange(0, f + 1, f / 5))  # 设置y轴格子线间隔为f/5
#
#
# if __name__ == '__main__':
#     imgs_root = './pic_out'  # 存储图片的子目录：本py文件同级下的imgs_scan文件夹
#
#
#     def get_save_path(img_name):
#         return imgs_root + '/' + img_name
#
#
#     scanname = ''
#     img = cv.imread(imgs_root + '/' + scanname + '.jpg')
#
#     # img = cv.imread(imgs_root + '/' + scanname + '.png')
#
#     # cv.imshow("origin", img)  # 给img取名origin
#
#     pair_pos_list, img, list_reverse, val_x, val_y = get_points_map_and_img(img)
#     print(list_reverse)
#
#     if is_need_save:
#         cv.imwrite(get_save_path('main_script_result.jpg'), img)
#
#
#     def pic_inches(pixel_width, pixel_height, dpi):
#         inches_width = pixel_width / dpi
#         inches_height = pixel_height / dpi
#         # 将像素尺寸转换为英寸
#         """
#         Matplotlib中，图像的尺寸是以英寸为单位指定的，而不是像素
#         可以通过调整图像的尺寸和分辨率（DPI）来达到特定的像素尺寸
#         像素尺寸 = 英寸尺寸 × 分辨率（DPI）
#
#         例如，创建一个1200x140像素的图像，选择一个尺寸比例（以英寸为单位）和一个DPI值，使得尺寸乘以DPI等于你想要的像素尺寸。假设我们选择了100
#         DPI，那么我们需要的图像尺寸应该是：
#         宽度 = 1200像素 / 100 DPI = 12英寸
#         高度 = 140像素 / 100 DPI = 1.4英寸
#         """
#         return inches_width, inches_height
#
#
#     cv.waitKey(0)
#     cv.destroyAllWindows()
