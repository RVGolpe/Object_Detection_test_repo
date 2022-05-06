import os
import shutil
import json
import csv
import convert_to_coco

print("Preparing Data")
convert_to_coco.train_data()
convert_to_coco.test_data()
print ("> Dataset converted to COCO Format")
print ("  > Re-arranging dataset files")
print ("    > Please wait....")
current_directory = os.getcwd()
path = os.path.join(current_directory, "drinks")
names = os.listdir(path)
Directory_names = ['train2017', 'val2017', 'annotations']
path2 = os.path.join(current_directory, 'dataset')
if (not os.path.exists(path2)):
    os.makedirs(path2)
else:
    shutil.rmtree(path2)
    os.makedirs(path2)


sub_paths = []
for i in range(len(Directory_names)):
    sub_paths.append(os.path.join(path2, Directory_names[i]))
for paths in sub_paths:
    if (not os.path.exists(paths)):
        os.makedirs(paths)
    else:
        shutil.rmtree(paths)
        os.makedirs(paths)


for files in names:
    if not ("._" in files):
        if ((".jpg" in files) and (0<=int(files[:7])<=1000)):
            shutil.copy(os.path.join(path,files), sub_paths[0])
        elif ((".jpg" in files) and (int(files[:7])>=10000)):
            shutil.copy(os.path.join(path, files), sub_paths[1])
        elif (".json" in files):
            shutil.copy(os.path.join(path, files), sub_paths[2])
    
print ("DATASET READY")

