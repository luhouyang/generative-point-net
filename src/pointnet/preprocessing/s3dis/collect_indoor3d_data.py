import os
import sys
from indoor3d_util import DATA_PATH, collect_point_label

BASE_DIR = 'D:/storage/s3dis'
ROOT_DIR = 'D:/storage/s3dis'
sys.path.append(BASE_DIR)

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'stanford_indoor3d')
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    try:
        elements = anno_path.split('/')
        front_path = elements[-3].split('\\')
        out_filename = front_path[-1]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy

        # CHECK FILE NAME & PATH IF ERROR
        # print(out_filename)
        collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    except:
        print('ERROR! CHECK FILE NAME & PATH IF ERROR')
