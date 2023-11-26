import os

folder_path = '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/dota_dior_3_7_noise/' 
before = "opendet_bnoise1e-2-dior37"
after = "opendet_bnoise1e-2_dota-dior37"

for filename in os.listdir(folder_path):
    old_name = filename
    if before in filename:
        new_name = filename.replace(before, after)
        os.rename(os.path.join(folder_path, old_name), 
                  os.path.join(folder_path, new_name))
        print(old_name, '->', new_name)
