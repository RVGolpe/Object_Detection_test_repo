import os
import shutil
import json
import csv
from pycocotools import mask as _mask
import torch



def segmentation_list(x, y):
    new_list = []
    for i in x:
        for j in y:
            new_list.append(i)
            new_list.append(j)
            y.pop(0)
            break
    return [new_list]
def segmentation_area(seg_list):
    def encode(bimask):
        if len(bimask.shape) == 3:
            return _mask.encode(bimask)
        elif len(bimask.shape) == 2:
            h, w = bimask.shape
            return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]
    w = 640
    h = 480
    # A = COCO.MaskApi.frPoly(torch.tensor(seg_list), h, w)
    # area = COCO.MaskApi.area(A)
    frPyObjects = _mask.frPyObjects
    Rs = frPyObjects(seg_list, h, w )
    return int(_mask.area(Rs))

    

def train_data ():
    instances_train_dict = {}
    ith_image_dict = {}
    ith_image_dict_list = []
    ith_annon_dict_list = []
    ith_annon_dict = {}
    current_directory = os.getcwd()
    path = os.path.join(current_directory, "drinks")
    with open(os.path.join(path, 'labels_train.csv')) as csv_file:
        read = csv.reader(csv_file, delimiter=",")
        rev = list(read)[:0:-1]
        big_list = []
        big_list_not_bygroup = []
        bilang = 0

        while (bilang<(len(rev)-1)):
            mini_list = []
            j = bilang
            while (rev[j][0]==rev[j+1][0]):
                mini_list.append(rev[j])
                j += 1
            if (not (rev[j][0]==rev[j+1][0])):
                mini_list.append(rev[j])
                bilang = j+1
                big_list.append(mini_list[::-1])
        mini_list = [rev[j+1]]
        big_list.append(mini_list)
        for batch in big_list:
            for box in batch:
                big_list_not_bygroup.append(box)


        f = open(os.path.join(path, 'segmentation_train.json'))
        data = json.load(f)
        _via_img_metadata_jpgnames = list(data['_via_img_metadata'])
        
        count = 0
        id_count = 1
        elements = 0
        bbox_sample = []
        for i in _via_img_metadata_jpgnames:    
            ith_image_dict_list.append({"id": count, "height": 640, "width": 480, "file_name": i[:11]})
            for item in data['_via_img_metadata'][i]['regions']:   
                if (elements>=len(big_list_not_bygroup)):
                    break;         
                if (i[:11] == big_list_not_bygroup[elements][0]):
                    width = (int(big_list_not_bygroup[elements][2])-int(big_list_not_bygroup[elements][1]))
                    height = (int(big_list_not_bygroup[elements][4])-int(big_list_not_bygroup[elements][3]))
                    seg_list = segmentation_list(item['shape_attributes']['all_points_x'], item['shape_attributes']['all_points_y'])
                    seg_area = segmentation_area(seg_list)
                    # print (f"{i[:11]} is equal to {big_list_not_bygroup[elements][0]}")
                    ith_annon_dict = {'id': id_count, 'category_id': int(item['region_attributes']['Name']), 'iscrowd': 0,
                'segmentation': seg_list, 'image_id': count, 
                'bbox': [int(big_list_not_bygroup[elements][1]), int(big_list_not_bygroup[elements][3]), width, height], 'area': seg_area}
                    bbox_sample.append(big_list_not_bygroup[elements][1:5])
                    ith_annon_dict_list.append(ith_annon_dict)
                    elements += 1
                    continue            
                elif (int(i[:7])<int(big_list_not_bygroup[elements][0][:7])):
                    # print (f"elements: {elements}")
                    # print (f"{i[:11]} is not equal to {big_list_not_bygroup[elements][0]}")
                    continue
                elif (int(i[:7])>int(big_list_not_bygroup[elements][0][:7])):
                    elements += 1
                    continue           
            count += 1
            id_count += 1        
        categories = []
        for i in range(1,4):
            categories.append({"supercategory": "drink", "id": i, "name": data["_via_attributes"]["region"]["Name"]["options"][str(i)]})

        
        f.close()
    instances_train_dict['images'] = ith_image_dict_list
    instances_train_dict['annotations'] = ith_annon_dict_list
    instances_train_dict['categories'] = categories
    instances_train_dict['info'] = {}
    instances_train_dict['licenses'] = {}
    json_object = json.dumps(instances_train_dict, indent = 0)
    with open(os.path.join(path, "instances_train2017.json"), "w") as output_file:
        output_file.write(json_object)

def test_data ():
    instances_train_dict = {}
    ith_image_dict = {}
    ith_image_dict_list = []
    ith_annon_dict_list = []
    ith_annon_dict = {}
    current_directory = os.getcwd()
    path = os.path.join(current_directory, "drinks")
    with open(os.path.join(path, 'labels_test.csv')) as csv_file_test:
        read = csv.reader(csv_file_test, delimiter=",")
        rev_test = list(read)[:0:-1]
        big_list = []
        big_list_not_bygroup = []
        bilang = 0

        while (bilang<(len(rev_test)-3)):
            mini_list = []
            j = bilang
            while (rev_test[j][0]==rev_test[j+1][0]):
                mini_list.append(rev_test[j])
                j += 1
            if (not (rev_test[j][0]==rev_test[j+1][0])):
                mini_list.append(rev_test[j])
                bilang = j+1
                big_list.append(mini_list[::-1])
        mini_list_1 = [rev_test[j+1]]
        mini_list_2 = [rev_test[j+2]]
        mini_list_3 = [rev_test[j+3]]
        big_list.append(mini_list_3)
        big_list.append(mini_list_2)
        big_list.append(mini_list_1)
        for batch in big_list:
            for box in batch:
                big_list_not_bygroup.append(box)
        # for line in big_list_not_bygroup:
        #     print (line)    

        f = open(os.path.join(path, 'segmentation_test.json'))
        data = json.load(f)
        _via_img_metadata_jpgnames = list(data['_via_img_metadata'])
        count = 0
        id_count = 1
        elements = 0
        bbox_sample = []
        for i in _via_img_metadata_jpgnames:    
            ith_image_dict_list.append({"id": count, "height": 640, "width": 480, "file_name": i[:11]})
            for item in data['_via_img_metadata'][i]['regions']:   
                if (elements>=len(big_list_not_bygroup)):
                    print("done")
                    break;         
                if (i[:11] == big_list_not_bygroup[elements][0]):
                    width = (int(big_list_not_bygroup[elements][2])-int(big_list_not_bygroup[elements][1]))
                    height = (int(big_list_not_bygroup[elements][4])-int(big_list_not_bygroup[elements][3]))
                    seg_list = segmentation_list(item['shape_attributes']['all_points_x'], item['shape_attributes']['all_points_y'])
                    seg_area = segmentation_area(seg_list)
                    # print (f"{i[:11]} is equal to {big_list_not_bygroup[elements][0]}")
                    ith_annon_dict = {'id': id_count, 'category_id': int(item['region_attributes']['Name']), 'iscrowd': 0,
                'segmentation': seg_list, 'image_id': count, 
                'bbox': [int(big_list_not_bygroup[elements][1]), int(big_list_not_bygroup[elements][3]), width, height], "area": seg_area}
                    bbox_sample.append(big_list_not_bygroup[elements][1:5])
                    ith_annon_dict_list.append(ith_annon_dict)
                    elements += 1
                    continue            
                elif (int(i[:7])<int(big_list_not_bygroup[elements][0][:7])):
                    # print (f"elements: {elements}")
                    # print (f"{i[:11]} is not equal to {big_list_not_bygroup[elements][0]}")
                    continue
                elif (int(i[:7])>int(big_list_not_bygroup[elements][0][:7])):
                    elements += 1
                    continue           
            count += 1   
            id_count += 1 

        categories = []
        for i in range(1,4):
            categories.append({"supercategory": "drink", "id": i, "name": data["_via_attributes"]["region"]["Name"]["options"][str(i)]})
          
        f.close()
    instances_train_dict['images'] = ith_image_dict_list
    instances_train_dict['annotations'] = ith_annon_dict_list
    instances_train_dict['categories'] = categories
    json_object = json.dumps(instances_train_dict, indent = 0)
    with open(os.path.join(path, "instances_val2017.json"), "w") as output_file:
        output_file.write(json_object)


train_data()
test_data()




# print(instances_train_dict)  




# id kang mask
# category id -- label kung juice etc.
# iscrowd
# segmentation
# image_id -- kung nasaan na image located yung mask
# area
# bbox







# print (_via_img_metadata_keys)






# with open('labels_train.csv') as csv_file:
#     read = csv.reader(csv_file, delimiter=",")

#     for i in read:
#         ith_image_dict['id': i[]]

# print(instances_train_dict)        














