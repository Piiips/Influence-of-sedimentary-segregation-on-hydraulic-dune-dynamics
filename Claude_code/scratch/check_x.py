import numpy as np

D = np.load('outputs/analisis/Single_slope_model_timeseries.npz')
station_keys = D['station_keys']
station_x = D['station_x']
c_mig = D['c_mig']
t_rec = D['t_rec']

print("c_mig:", c_mig)
print("station_x (comoving):", station_x)

