import numpy as np

D = np.load('outputs/analisis/Single_slope_model_timeseries.npz')
phi_st = D['phi_st']
station_keys = D['station_keys']

idx25 = np.where(station_keys == 'lee25')[0][0]
idx50 = np.where(station_keys == 'lee50')[0][0]
idx75 = np.where(station_keys == 'lee75')[0][0]

for name, idx in [('25%', idx25), ('50%', idx50), ('75%', idx75)]:
    phi_bot25 = np.mean(phi_st[:, idx, :10], axis=1) # 10/40 = 25%
    print(f"{name} Bot 25% mean:", np.mean(phi_bot25))

