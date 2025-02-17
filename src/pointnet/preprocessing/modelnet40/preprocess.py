import trimesh
"""
@software{trimesh,
	author = {{Dawson-Haggerty et al.}},
	title = {trimesh},
	url = {https://trimesh.org/},
	version = {3.2.0},
	date = {2019-12-8},
}
"""

import os

# root dir of dataset
root_dir = "D:/storage/ModelNet40"


def convert_off_to_ply(root_dir, split="train"):
    """
    Convert all .off files in the ModelNet40 dataset to .ply format.
    
    Parameters:
    - root_dir (str): Path to the ModelNet40 dataset.
    - split (str): 'train' or 'test', depending on which dataset to convert.
    """
    # train_txt_path = os.path.join(root_dir, "train.txt")

    # with open(train_txt_path, "r") as f:
    #     class_names = [line.strip() for line in f.readlines()]
    
    class_names = [
        "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
        "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
        "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
        "mantel", "monitor", "night_stand", "person", "piano", "plant",
        "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
        "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"
    ]

    for class_name in class_names:
        input_folder = os.path.join(root_dir, class_name, split)
        output_folder = os.path.join(root_dir, class_name, f"{split}_ply")

        if not os.path.exists(input_folder):
            print(f"Skipping {input_folder}, folder not found.")
            continue

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in os.listdir(input_folder):
            if file_name.endswith(".off"):
                input_file = os.path.join(input_folder, file_name)
                output_file = os.path.join(output_folder,
                                           file_name.replace(".off", ".ply"))

                try:
                    mesh = trimesh.load_mesh(input_file, file_type='off')
                    mesh.export(output_file, file_type='ply')
                    print(f"Converted: {input_file} → {output_file}")
                except Exception as e:
                    print(f"Failed to convert {input_file}: {e}")


convert_off_to_ply(root_dir, "train")
convert_off_to_ply(root_dir, "test")
