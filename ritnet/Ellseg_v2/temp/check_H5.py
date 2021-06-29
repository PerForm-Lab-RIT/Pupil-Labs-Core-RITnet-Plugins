import os
import h5py

path_H5 = '/scratch/multiset/All'

list_H5 = ['Fuhl_data set new V_28', 'LPW_8_7_34', 'NVGaze_nvgaze_female_02_public_50K_4_30', 'Santini_Santini_1_1', 'Swirski_p1-right_3', 'UnityEyes_0_0', 'riteyes-s-general_17_22', 'riteyes-s-natural_24_17', 'Fuhl_data set XVIII_17', 'LPW_3_21_0', 'NVGaze_nvgaze_male_03_public_50K_3_33', 'Santini_Santini_4_13', 'Swirski_p1-right_3', 'UnityEyes_4_4', 'riteyes-s-general_20_3', 'riteyes-s-natural_7_2', 'Fuhl_data set XX_18', 'LPW_13_1_17', 'NVGaze_nvgaze_female_04_public_50K_4_1', 'Santini_Santini_4_12', 'Swirski_p2-left_2', 'UnityEyes_7_7', 'riteyes-s-general_17_22', 'riteyes-s-natural_16_16', 'Fuhl_data set X_9', 'LPW_5_10_25', 'NVGaze_nvgaze_male_01_public_50K_1_9', 'Santini_Santini_1_0', 'Swirski_p2-right_0', 'UnityEyes_2_2', 'riteyes-s-general_9_14', 'riteyes-s-natural_24_17']

sample_ID = [1000,  165,  270,  469,  103, 1343,  209,   72, 4461, 1853,  374,  243,
             56,  644, 1315, 1349, 1318,  505,  452,  708,   42, 1167,  527, 1862,
             593, 1457,  401,  347,   44, 1356, 1418,  157]
             
for idx, H5_file in enumerate(list_H5):
    path_file = os.path.join(path_H5, H5_file)
    H5_obj = h5py.File(path_file+'.h5', 'r')
   
        
    loc = sample_ID[idx]
  
    print('--------------')      
    print(H5_file)
        
    print('Pupil Location')
    print(H5_obj['pupil_loc'][loc])
        
    print('Pupil Ellipse')
    if H5_obj['Fits']['pupil'].__len__()!=0:
        print(H5_obj['Fits']['pupil'][loc])
        
    print('Iris Ellipse')
    if H5_obj['Fits']['iris'].__len__()!=0:
        print(H5_obj['Fits']['iris'][loc])
        
    print('Mask')
    if H5_obj['Masks_noSkin'].__len__()!=0:
        print(H5_obj['Masks_noSkin'][loc].shape)
    
    H5_obj.close()
