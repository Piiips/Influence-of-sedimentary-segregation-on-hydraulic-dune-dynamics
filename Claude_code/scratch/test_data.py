import numpy as np

D = np.load('outputs/analisis/Single_slope_model_timeseries.npz')
print("Keys:", D.files)
print("station_keys:", D['station_keys'])
print("t_rec shape:", D['t_rec'].shape)
print("phi_st shape:", D['phi_st'].shape)
print("ec shape:", D['ec'].shape)
