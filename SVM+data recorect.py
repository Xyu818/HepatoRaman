import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# === 美化设置 ===
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'Arial'
})

# === 用户配置 ===
glyco_path       = r"L:\微软浏览器下载\xy\Datas\AI practice\2.12(1)\glycoprotein.xlsx"
afp_path         = r"L:\微软浏览器下载\xy\Datas\AI practice\2.12(1)\AFP.xlsx"
raw_path         = r"L:\微软浏览器下载\xy\Datas\mine\20250210\20250210\i82_Copy.txt"
model_path       = r"L:\办公软件\微信\WeChat\WeChat Files\wxid_npk4532ue7eu22\FileStorage\File\2025-02\2.12\svm_model.pkl"
wave_of_interest = 500.0

# === 1. 加载模型 & 波数网格 ===
pipeline    = joblib.load(model_path)
train_waves = pd.read_excel(glyco_path).iloc[:, 0].astype(float).values

# === 2. 读取原始拉曼成像数据，强制转成数值 ===
raw_df = pd.read_csv(
    raw_path,
    sep=r'\s+',
    names=['X','Y','Wave','Intensity'],
    comment='#',
    header=None,
    low_memory=False
)

for col in ['X','Y','Wave','Intensity']:
    raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')

raw_df = raw_df.dropna(subset=['X','Y','Wave','Intensity'])

unique_x = np.sort(raw_df['X'].unique())
unique_y = np.sort(raw_df['Y'].unique())
nx, ny   = len(unique_x), len(unique_y)

# === 3a. 构建500 cm⁻¹ 强度图 ===
img_raw = np.zeros((ny, nx))
for i, yv in enumerate(unique_y):
    for j, xv in enumerate(unique_x):
        grp = raw_df[(raw_df['X']==xv)&(raw_df['Y']==yv)][['Wave','Intensity']].values
        if grp.size == 0:
            img_raw[i,j] = np.nan
        else:
            idx = np.argmin(np.abs(grp[:,0] - wave_of_interest))
            img_raw[i,j] = grp[idx,1]

# === 3b. 构建1035 cm⁻¹ 强度图 ===
wave_1035 = 1035.0
img_1035 = np.zeros((ny, nx))
for i, yv in enumerate(unique_y):
    for j, xv in enumerate(unique_x):
        grp = raw_df[(raw_df['X']==xv)&(raw_df['Y']==yv)][['Wave','Intensity']].values
        if grp.size == 0:
            img_1035[i,j] = np.nan
        else:
            idx = np.argmin(np.abs(grp[:,0] - wave_1035))
            img_1035[i,j] = grp[idx,1]

# === 4. 构建全谱预测矩阵 ===
grouped = raw_df.groupby(['X','Y'])
n_pix   = nx * ny
X_raw   = np.zeros((n_pix, train_waves.size))
coords  = []
for idx, ((xv, yv), grp) in enumerate(grouped):
    spec = grp[['Wave','Intensity']].sort_values('Wave').values
    w, I = spec[:,0], spec[:,1]
    X_raw[idx, :] = np.interp(train_waves, w, I)
    coords.append((xv, yv))

# === 5. 用已有模型预测 AFP 类别（0或1） ===
labels = pipeline.predict(X_raw).reshape(ny, nx)

# === 6. 分类标签为1的像素放大，其余不变 ===
factor = np.where(labels == 1, 2.0, 1.0)
img_corr = img_raw * factor

# === 7. 可视化（改良颜色+美化字体） ===
fig, axs = plt.subplots(2, 2, figsize=(12,10))

# 500 cm⁻¹ 原始图 —— 红橙黄色
im0 = axs[0, 0].imshow(
    img_raw, origin='lower',
    extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
    cmap='hot'
)
axs[0, 0].set_title(f'Raw @ {wave_of_interest} cm$^{{-1}}$')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
plt.colorbar(im0, ax=axs[0, 0], label='Intensity')

# 1035 cm⁻¹ 原始图 —— cividis（蓝绿，清晰）
im1 = axs[0, 1].imshow(
    img_1035, origin='lower',
    extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
    cmap='cividis'
)
axs[0, 1].set_title('Raw @ 1035 cm$^{-1}$')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
plt.colorbar(im1, ax=axs[0, 1], label='Intensity')

# 预测类别 —— 绿色
im2 = axs[1, 0].imshow(
    labels, origin='lower',
    extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
    cmap='Greens', vmin=0, vmax=1
)
axs[1, 0].set_title('Predicted Class (AFP=1)')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
plt.colorbar(im2, ax=axs[1, 0], label='Class')

# 矫正后强度图 —— 红橙黄色
im3 = axs[1, 1].imshow(
    img_corr, origin='lower',
    extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
    cmap='hot'
)
axs[1, 1].set_title('Corrected (AFP=1 boosted)')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')
plt.colorbar(im3, ax=axs[1, 1], label='Corrected Intensity')

plt.tight_layout()
plt.show()

# === 8. 导出每像素数据 ===
df_out = pd.DataFrame(coords, columns=['X','Y'])
df_out['Raw_500']    = img_raw.flatten()
df_out['Raw_1035']   = img_1035.flatten()
df_out['Class']      = labels.flatten()
df_out['Corrected']  = img_corr.flatten()
df_out.to_csv('raman_correction_results.csv', index=False)
print("Results saved to raman_correction_results.csv")
