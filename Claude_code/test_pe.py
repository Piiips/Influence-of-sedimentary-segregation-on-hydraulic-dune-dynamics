import numpy as np
import Single_slope_model as M

data = np.load('outputs/analisis/Single_slope_model_timeseries.npz', allow_pickle=True)
phi_st = data['phi_st']
st_keys = data['station_keys']
st_idx = data['station_idx']
h_all = data['h']
ec = data['ec']

for i, key in enumerate(st_keys):
    if key in ['lee25', 'lee50', 'lee75']:
        h_st = h_all[st_idx[i]]
        phi = phi_st[:, i, :] # (time, z)
        # p_c is shape (Nz)
        p_c = M.nu_pack * h_st * (1 - ec) + M.p_floor
        # dbar is (time, z)
        dbar = (1 - phi) * M.d_l + phi * M.d_s
        Ffac = (M.R - 1) + M.E_seg * (1 - phi) * (M.R - 1)**2
        # Pe is (time, z)
        Pe = (M.B_seg / M.A_diff) * Ffac * h_st / (M.C_seg * dbar + p_c)
        
        print(f"Station {key}: h = {h_st*1000:.2f} mm")
        print(f"  Pe mean at eta=0.0: {Pe[:, 0].mean():.2f}")
        print(f"  Pe mean at eta=0.5: {Pe[:, 20].mean():.2f}")
        print(f"  Pe mean at eta=1.0: {Pe[:, -1].mean():.2f}")

