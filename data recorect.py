import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============== 1. 读入数据 ==============
# 假设你的数据文件名是 'data.txt'，并且以空白字符分隔
# names 参数显式指定列名，如果文本文件中已经有表头，可以去掉 names=... 或自行调整
data = pd.read_csv(
    r"L:\微软浏览器下载\xy\Datas\mine\20250210\20250210\i82_Copy.txt",        # 替换为你的实际文件路径
    sep=r'\s+',        # 以空格或制表符为分隔
    names=['X','Y','Wave','Intensity'],
    comment='#',       # 忽略以 '#' 开头的注释行
    header=None        # 若文件本身无标题行，可指定 header=None；若有则相应调整
)

# ============== 2. 检查数据结构 ==============
# 打印前几行，验证列名和数据是否正确
print(data.head())

# ============== 3. 获取唯一的坐标值并排序 ==============
unique_x = np.sort(data['X'].unique())
unique_y = np.sort(data['Y'].unique())
nx = len(unique_x)
ny = len(unique_y)
print(f"Unique X count: {nx}, Unique Y count: {ny}")

# ============== 4. 按 (X, Y) 分组，存储每个点的 (Wave, Intensity) 信息 ==============
grouped = data.groupby(['X', 'Y'])

# 字典结构： spec_dict[(x_val, y_val)] = [ [wave1, I1], [wave2, I2], ... ]
spec_dict = {}
for (x_val, y_val), group_df in grouped:
    # 只保留 Wave 和 Intensity 两列，并转成 NumPy 数组，方便后续处理
    wave_int = group_df[['Wave', 'Intensity']].values
    spec_dict[(x_val, y_val)] = wave_int

# ============== 5. 选择波数并构造二维图像 ==============
# 这里示例：选定某个“目标波数”做成像，取最接近该波数的强度
# 你也可以改为对某个波数范围进行积分
wave_of_interest = 500.0  # 根据需要修改

# 初始化空白图像 (ny 行, nx 列)
image = np.zeros((ny, nx))

# 双重循环：依次访问每个 (x, y)，在 spec_dict 中找到与 wave_of_interest 最接近的强度
for i, yv in enumerate(unique_y):
    for j, xv in enumerate(unique_x):
        wave_int = spec_dict[(xv, yv)]
        # wave_int 是一个二维数组，每行 [Wave, Intensity]
        # 1) 找到最接近 wave_of_interest 的行索引
        idx = np.argmin(np.abs(wave_int[:, 0] - wave_of_interest))
        intensity_val = wave_int[idx, 1]
        # 2) 放进图像矩阵
        image[i, j] = intensity_val

# ============== 6. 显示并保存图像 ==============
# extent 设置坐标轴范围 (xmin, xmax, ymin, ymax)，origin='lower' 让 (y 最小) 在下方
plt.figure(figsize=(6, 5))
plt.imshow(
    image,
    extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
    origin='lower',
    cmap='inferno'
)
plt.colorbar(label='Intensity')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Raman intensity around {wave_of_interest} cm^-1')
plt.tight_layout()
plt.show()
