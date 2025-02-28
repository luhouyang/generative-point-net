import os
import numpy as np
import open3d as o3d


def fix_off_file(file_path):
    """Ensures the OFF file has the correct format."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # If the first line is not exactly "OFF", modify it
    if not lines[0].strip() == "OFF":
        lines[0] = "OFF\n" + lines[0][3:]  # Insert a newline after "OFF"

    with open(file_path, 'w') as f:
        f.writelines(lines)


def convert_off_to_txt(root_dir, split="train"):
    """
    Convert all .off files in the ModelNet40 dataset to .txt format.
    
    Parameters:
    - root_dir (str): Path to the ModelNet40 dataset.
    - split (str): 'train' or 'test', depending on which dataset to convert.
    """
    class_names = [
        "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
        "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
        "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
        "mantel", "monitor", "night_stand", "person", "piano", "plant",
        "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
        "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
    ]

    failed = 0
    failed_class = []

    for class_name in class_names:
        input_folder = os.path.join(root_dir, class_name, split)
        output_folder = os.path.join(root_dir, class_name, f"{split}_txt")

        if not os.path.exists(input_folder):
            print(f"Skipping {input_folder}, folder not found.")
            continue

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in os.listdir(input_folder):
            if file_name.endswith(".off"):
                input_file = os.path.join(input_folder, file_name)
                output_file = os.path.join(output_folder,
                                           file_name.replace(".off", ".txt"))

                try:
                    # Fix OFF file formatting before loading
                    fix_off_file(input_file)

                    # Load the mesh after fixing
                    mesh = o3d.io.read_triangle_mesh(input_file)
                    points = np.asarray(mesh.vertices)

                    # Save vertices to .txt
                    with open(output_file, "w") as f:
                        for point in points:
                            f.write(f"{point[0]} {point[1]} {point[2]}\n")

                    print(f"Converted: {input_file} -> {output_file}")
                except Exception as e:
                    failed += 1
                    if class_name not in failed_class:
                        failed_class.append(class_name)
                    print(f"Failed to convert {input_file}: {e}")

    print(f"FAILED: {failed}")

    return failed, failed_class


root_dir = "D:/storage/ModelNet40"
train_failed, train_failed_class = convert_off_to_txt(root_dir, "train")
test_failed, test_failed_class = convert_off_to_txt(root_dir, "test")

print(f"Train | FAILED: {train_failed} | FAILED_CLASS: {train_failed_class}")
print(f"Test | FAILED: {test_failed} | FAILED_CLASS: {test_failed_class}")

# import trimesh
# """
# @software{trimesh,
# 	author = {{Dawson-Haggerty et al.}},
# 	title = {trimesh},
# 	url = {https://trimesh.org/},
# 	version = {3.2.0},
# 	date = {2019-12-8},
# }
# """

# import os

# # root dir of dataset
# root_dir = "D:/storage/ModelNet40"

# def convert_off_to_ply(root_dir, split="train"):
#     """
#     Convert all .off files in the ModelNet40 dataset to .ply format.

#     Parameters:
#     - root_dir (str): Path to the ModelNet40 dataset.
#     - split (str): 'train' or 'test', depending on which dataset to convert.
#     """
#     # train_txt_path = os.path.join(root_dir, "train.txt")

#     # with open(train_txt_path, "r") as f:
#     #     class_names = [line.strip() for line in f.readlines()]

#     class_names = [
#         "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
#         "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
#         "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
#         "mantel", "monitor", "night_stand", "person", "piano", "plant",
#         "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
#         "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
#     ]

#     for class_name in class_names:
#         input_folder = os.path.join(root_dir, class_name, split)
#         output_folder = os.path.join(root_dir, class_name, f"{split}_ply")

#         if not os.path.exists(input_folder):
#             print(f"Skipping {input_folder}, folder not found.")
#             continue

#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)

#         for file_name in os.listdir(input_folder):
#             if file_name.endswith(".off"):
#                 input_file = os.path.join(input_folder, file_name)
#                 output_file = os.path.join(output_folder,
#                                            file_name.replace(".off", ".ply"))

#                 try:
#                     mesh = trimesh.load_mesh(input_file, file_type='off')
#                     mesh.export(output_file, file_type='ply')
#                     print(f"Converted: {input_file} → {output_file}")
#                 except Exception as e:
#                     print(f"Failed to convert {input_file}: {e}")

# # convert_off_to_ply(root_dir, "train")
# # convert_off_to_ply(root_dir, "test")
