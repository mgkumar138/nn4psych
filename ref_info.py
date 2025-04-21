'''
storage for data_dir, parameter info

-#to-do - setup for when running models initially
- use for setting up pkls in get_behavior.py
- also getting out pkls in behav_figures.py
'''



def ref_info(version:str = None):
    """
    # Define the reference information for each parameter
    # Returns a dictionary with values and file pattern for the specified parameter
    -depends on the different versions (seems like 3 and 5 right now)
    """
    if version == "V3":
        ref_info_dict = {
            "gamma": {
                "values": [0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1],
                "file_pattern": "*_V3_{val}g_0.0rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
            },
            "rollout": {
                "values": [5, 10, 20, 30, 50, 75, 100, 150, 200],
                "file_pattern": "*_V3_0.95g_0.0rm_{val}bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
            },
            "preset": {
                "values": [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                "file_pattern": "*_V3_0.95g_{val}rm_100bz_0.0td_1.0tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
            },
            "scale": {
                "values": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
                "file_pattern": "*_V3_0.95g_0.0rm_100bz_0.0td_{val}tds_Nonelb_Noneup_64n_50000e_10md_5.0rz_*s.pth"
            }
        }
        data_dir = "./model_params_101000/"

    elif version == "V5":
        ref_info_dict = {
            "gamma": {
                "values": [0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.1],
                "file_pattern": "*_V5_{val}g_0.0rm_50bz_0.0td_1.0tds_64n_50000e_10md_5.0rz_*s.pth"
            },
            "rollout": {
                "values": [5, 10, 20, 30, 40, 50, 75, 100, 150, 200],
                "file_pattern": "*_V5_0.95g_0.0rm_{val}bz_0.0td_1.0tds_64n_50000e_10md_5.0rz_*s.pth"
            },
            "preset": {
                "values": [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                "file_pattern": "*_V5_0.95g_{val}rm_50bz_0.0td_1.0tds_64n_50000e_10md_5.0rz_*s.pth"
            },
            "scale": {
                "values": [1e-4, 1e-3],
                "file_pattern": "*_V5_0.95g_{val}rm_*bz_*tds_Nonelb_Noneup_*n_*e_*md_*rz_*s.pth"
            }
        }
        data_dir = "./model_params_2/"
    else: 
        raise ValueError("Invalid version. Choose either 'V3' or 'V5'.")
    
    return data_dir, ref_info_dict


#from analyze_hyperparams originally 

data_dir = "./model_params_2/"
bias = False

