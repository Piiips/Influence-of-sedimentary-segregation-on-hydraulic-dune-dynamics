import numpy as np
import matplotlib.pyplot as plt

D = np.load('outputs/analisis/Single_slope_model_timeseries.npz')
phi_st = D['phi_st']
t_rec = D['t_rec']
station_keys = D['station_keys']

idx25 = np.where(station_keys == 'lee25')[0][0]
# Use column average
phi_25 = np.mean(phi_st[:, idx25, :], axis=1)

print("Std dev:", np.std(phi_25))
print("Min/Max:", np.min(phi_25), np.max(phi_25))
