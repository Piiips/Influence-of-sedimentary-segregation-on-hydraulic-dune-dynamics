import numpy as np

D = np.load('outputs/analisis/Single_slope_model_timeseries.npz')
phi_st = D['phi_st']
station_keys = D['station_keys']

idx25 = np.where(station_keys == 'lee25')[0][0]
idx50 = np.where(station_keys == 'lee50')[0][0]
idx75 = np.where(station_keys == 'lee75')[0][0]

# Check depth-averaged
phi_mean_25 = np.mean(phi_st[:, idx25, :], axis=1)
# Check surface (eta ~ 1, which is the last index in ec)
phi_surf_25 = phi_st[:, idx25, -1]
# Check bottom (eta ~ 0)
phi_bot_25 = phi_st[:, idx25, 0]

print("Mean range:", np.min(phi_mean_25), np.max(phi_mean_25))
print("Surf range:", np.min(phi_surf_25), np.max(phi_surf_25))
print("Bot range:", np.min(phi_bot_25), np.max(phi_bot_25))
