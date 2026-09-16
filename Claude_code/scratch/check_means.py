import numpy as np

D = np.load('outputs/analisis/Single_slope_model_timeseries.npz')
phi_st = D['phi_st']
station_keys = D['station_keys']

idx25 = np.where(station_keys == 'lee25')[0][0]
idx50 = np.where(station_keys == 'lee50')[0][0]
idx75 = np.where(station_keys == 'lee75')[0][0]

for name, idx in [('25%', idx25), ('50%', idx50), ('75%', idx75)]:
    phi_bot = phi_st[:, idx, 0] # eta=0
    print(f"{name} Bot mean:", np.mean(phi_bot))
    phi_mean = np.mean(phi_st[:, idx, :], axis=1)
    print(f"{name} Col mean:", np.mean(phi_mean))
    # average bottom 10%
    phi_bot10 = np.mean(phi_st[:, idx, :4], axis=1)
    print(f"{name} Bot 10% mean:", np.mean(phi_bot10))
    # surface
    phi_surf = phi_st[:, idx, -1]
    print(f"{name} Surf mean:", np.mean(phi_surf))
    print("---")
