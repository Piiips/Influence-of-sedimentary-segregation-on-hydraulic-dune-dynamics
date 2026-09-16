import numpy as np
D = np.load('outputs/analisis/Single_slope_model_timeseries.npz')
phi_st = D['phi_st']
station_keys = D['station_keys']
idx50 = np.where(station_keys == 'lee50')[0][0]
phi_bot = phi_st[:, idx50, 0]
print("Std dev at bottom (eta=0):", np.std(phi_bot))
print("Max at bottom:", np.max(phi_bot))
print("Min at bottom:", np.min(phi_bot))

phi_avg = np.mean(phi_st[:, idx50, :], axis=1)
print("Std dev (column mean):", np.std(phi_avg))
print("Max (column mean):", np.max(phi_avg))
print("Min (column mean):", np.min(phi_avg))
