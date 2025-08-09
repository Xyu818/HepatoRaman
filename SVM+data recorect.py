import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib


plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'font.family': 'Arial'
})


glyco_path       = 
afp_path         = 
raw_path         = 
model_path       = 
wave_of_interest = 500.0


pipeline    = joblib.load(model_path)
train_waves = pd.read_excel(glyco_path).iloc[:, 0].astype(float).values


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


img_raw = np.zeros((ny, nx))
for i, yv in enumerate(unique_y):
    for j, xv in enumerate(unique_x):
        grp = raw_df[(raw_df['X']==xv)&(raw_df['Y']==yv)][['Wave','Intensity']].values
        if grp.size == 0:
            img_raw[i,j] = np.nan
        else:
            idx = np.argmin(np.abs(grp[:,0] - wave_of_interest))
            img_raw[i,j] = grp[idx,1]


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


grouped = raw_df.groupby(['X','Y'])
n_pix   = nx * ny
X_raw   = np.zeros((n_pix, train_waves.size))
coords  = []
for idx, ((xv, yv), grp) in enumerate(grouped):
    spec = grp[['Wave','Intensity']].sort_values('Wave').values
    w, I = spec[:,0], spec[:,1]
    X_raw[idx, :] = np.interp(train_waves, w, I)
    coords.append((xv, yv))


labels = pipeline.predict(X_raw).reshape(ny, nx)


factor = np.where(labels == 1, 2.0, 1.0)
img_corr = img_raw * factor


fig, axs = plt.subplots(2, 2, figsize=(12,10))


im0 = axs[0, 0].imshow(
    img_raw, origin='lower',
    extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
    cmap='hot'
)
axs[0, 0].set_title(f'Raw @ {wave_of_interest} cm$^{{-1}}$')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
plt.colorbar(im0, ax=axs[0, 0], label='Intensity')


im1 = axs[0, 1].imshow(
    img_1035, origin='lower',
    extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
    cmap='cividis'
)
axs[0, 1].set_title('Raw @ 1035 cm$^{-1}$')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
plt.colorbar(im1, ax=axs[0, 1], label='Intensity')


im2 = axs[1, 0].imshow(
    labels, origin='lower',
    extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
    cmap='Greens', vmin=0, vmax=1
)
axs[1, 0].set_title('Predicted Class (AFP=1)')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')
plt.colorbar(im2, ax=axs[1, 0], label='Class')


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


df_out = pd.DataFrame(coords, columns=['X','Y'])
df_out['Raw_500']    = img_raw.flatten()
df_out['Raw_1035']   = img_1035.flatten()
df_out['Class']      = labels.flatten()
df_out['Corrected']  = img_corr.flatten()
df_out.to_csv('raman_correction_results.csv', index=False)
print("Results saved to raman_correction_results.csv")

