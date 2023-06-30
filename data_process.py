import os 
import shutil

path = 'data/custom/pikaqiu_out/images'
files = os.listdir(path)

for index, file in enumerate(files, start=1):
    if file.endswith('.png'):
        src = os.path.join(path, file)
        dst = os.path.join(path, str(index) + '.png')
        shutil.move(src, dst)