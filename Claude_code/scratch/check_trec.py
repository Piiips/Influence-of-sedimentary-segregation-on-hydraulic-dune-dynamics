import numpy as np
D = np.load('outputs/analisis/Single_slope_model_timeseries.npz')
t_rec = D['t_rec']
print("t_rec shape:", t_rec.shape)
print("t_rec min, max:", t_rec.min(), t_rec.max())
print("t_rec step:", t_rec[1]-t_rec[0])
