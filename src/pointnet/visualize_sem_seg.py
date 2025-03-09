import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
import argparse


def load_segmentation_labels(txt_path):
    with open(txt_path, 'r') as f:
        labels = [int(x.strip()) for x in f.readlines() if x.strip()]
    return np.array(labels)


def load_point_cloud_data(npy_path):
    data = np.load(npy_path)
    points = data[:, :3]
    return data, points


def setup_s3dis_metadata():
    g_classes = [
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
    ]

    g_class2label = {cls: i for i, cls in enumerate(g_classes)}

    g_class2color = {
        'ceiling': [0, 255, 0],
        'floor': [0, 0, 255],
        'wall': [0, 255, 255],
        'beam': [255, 255, 0],
        'column': [255, 0, 255],
        'window': [100, 100, 255],
        'door': [200, 200, 100],
        'table': [170, 120, 200],
        'chair': [255, 0, 0],
        'sofa': [200, 100, 100],
        'bookcase': [10, 200, 100],
        'board': [200, 200, 200],
        'clutter': [50, 50, 50]
    }

    g_label2color = {
        g_class2label[cls]: g_class2color[cls]
        for cls in g_classes
    }
    return g_classes, g_class2label, g_class2color, g_label2color


def create_colored_point_cloud(points, labels, label2color):
    colors = np.array(
        [label2color.get(label, [100, 100, 100]) for label in labels]) / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_point_cloud(pcd,
                          window_title="Semantic Segmentation Visualization"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title)
    vis.add_geometry(pcd)

    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0

    # Set camera position
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    vis.run()
    vis.destroy_window()


def create_label_visualization(g_classes, g_label2color, output_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cls_name in enumerate(g_classes):
        color = np.array(g_label2color[i]) / 255.0
        ax.bar(i, 1, color=color)

    ax.set_xticks(range(len(g_classes)))
    ax.set_xticklabels(g_classes, rotation=45, ha='right')
    ax.set_title('Semantic Classes and Colors')
    ax.set_ylabel('Legend')
    ax.set_yticks([])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Legend saved to {output_path}")
    else:
        plt.show()

    plt.close()


def save_colored_point_cloud(pcd, output_path):
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Colored point cloud saved to {output_path}")


def process_scene(data_dir,
                  label_dir,
                  scene_name,
                  output_dir=None,
                  create_legend=False):
    # Setup metadata
    g_classes, g_class2label, g_class2color, g_label2color = setup_s3dis_metadata(
    )

    # Construct file paths
    npy_path = os.path.join(data_dir, f"{scene_name}.npy")
    label_path = os.path.join(label_dir, f"{scene_name}.txt")

    # Check if files exist
    if not os.path.exists(npy_path):
        print(f"Error no data {npy_path}")
        return False

    if not os.path.exists(label_path):
        print(f"Error file not found {label_path}")
        return False

    data, points = load_point_cloud_data(npy_path)

    labels = load_segmentation_labels(label_path)

    if len(labels) != len(points):
        print(
            f"Number of labels ({len(labels)}) doesn't match number of points ({len(points)})"
        )
        labels = labels[:len(points)] if len(labels) > len(points) else np.pad(
            labels, (0, len(points) - len(labels)))

    pcd = create_colored_point_cloud(points, labels, g_label2color)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ply_path = os.path.join(output_dir, f"{scene_name}_colored.ply")
        save_colored_point_cloud(pcd, ply_path)

        if create_legend:
            legend_path = os.path.join(output_dir, "color_legend.png")
            create_label_visualization(g_classes, g_label2color, legend_path)

    window_title = f"Semantic Segmentation: {scene_name}"
    visualize_point_cloud(pcd, window_title)

    return True


def process_all_scenes(data_dir,
                       label_dir,
                       output_dir=None,
                       create_legend=False):
    data_files = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.npy')]

    label_files = [f[:-4] for f in os.listdir(label_dir) if f.endswith('.txt')]

    common_scenes = list(set(data_files).intersection(set(label_files)))

    if not common_scenes:
        print("Error: No matching data and label files found.")
        return

    print(f"Found {len(common_scenes)} scenes with both data and label files.")

    for scene_name in common_scenes:
        print(f"Processing scene: {scene_name}")
        process_scene(data_dir, label_dir, scene_name, output_dir,
                      create_legend)

        # Create legend only once
        create_legend = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='dir containing .npy files')
    parser.add_argument('--label_dir',
                        type=str,
                        required=True,
                        help='dir containing segmentation label .txt files')
    parser.add_argument(
        '--scene_name',
        type=str,
        default=None,
        help='provide only one scene to process, without file extensions')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='dir to save colored point clouds and legend')
    parser.add_argument('--create_legend',
                        action='store_true',
                        help='save output')

    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.scene_name:
        process_scene(args.data_dir, args.label_dir, args.scene_name,
                      args.output_dir, args.create_legend)
    else:
        process_all_scenes(args.data_dir, args.label_dir, args.output_dir,
                           args.create_legend)


if __name__ == "__main__":
    main()
