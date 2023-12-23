import os

folder_path = '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/projects/OpenRS/opendet2/configs/fair1m_mar20/' 
before = "mar20_crop512"
after = "mar20_crop1024"

for filename in os.listdir(folder_path):
    old_name = filename
    if before in filename:
        new_name = filename.replace(before, after)
        os.rename(os.path.join(folder_path, old_name), 
                  os.path.join(folder_path, new_name))
        print(old_name, '->', new_name)
