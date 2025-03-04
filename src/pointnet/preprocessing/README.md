## **Datasets**

- **ModelNet40** | [paper](https://arxiv.org/abs/1406.5670) | [dataset](https://3dshapenets.cs.princeton.edu) or [alternative source](https://modelnet.cs.princeton.edu)

    1. Download the ModelNet40 dataset & unzip
    1. Run preprocessing script [/src/pointnet/preprocessing/modelnet40/preprocessing.py](/src/pointnet/preprocessing/modelnet40/preprocess.py)
    1. Example testing command

        ```
        cd PATH\generative-point-net
        python -m src.test.dataset_test modelnet40 PATH\ModelNet40 
        ```

- **ModelNet10** | [paper](https://arxiv.org/abs/1406.5670) | [dataset](https://3dshapenets.cs.princeton.edu) or [alternative source](https://modelnet.cs.princeton.edu)

    1. Download the ModelNet10 dataset & unzip
    1. Delete `__MACOSX` directory
    1. Delete `raw` directory
    1. Delete all `.DS_Store` files
    1. Run preprocessing script [/src/pointnet/preprocessing/modelnet10/preprocessing.py](/src/pointnet/preprocessing/modelnet10/preprocess.py)
    1. Example testing command

        ```
        cd PATH\generative-point-net
        python -m src.test.dataset_test modelnet10 PATH\ModelNet10 
        ```

- **ShapeNetCore** | [paper](https://arxiv.org/abs/1512.03012) | [dataset](https://www.kaggle.com/datasets/mitkir/shapenet)

    1. Download the ShapeNetCore dataset & unzip
    1. Example testing command

        ```
        cd PATH\generative-point-net
        python -m src.test.dataset_test shapenet PATH\shapenet\shapenetcore_partanno_segmentation_benchmark_v0_normal 
        ```

- **Stanford Large-Scale Indoor Spaces 3D Dataset (S3DIS)** | [paper](https://ieeexplore.ieee.org/document/7780539) | [dataset](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform)

    1. Download Stanford3dDataset_v1.2_Aligned_Version.zip & unzip
    1. Copy [/src/pointnet/preprocessing/s3dis/meta](/src/pointnet/preprocessing/s3dis/meta) to the same directory as dataset
    1. Edit the `BASE_DIR` & `ROOT_DIR` in preprocessing script [/src/pointnet/preprocessing/s3dis/collect_indoor3d_data.py](/src/pointnet/preprocessing/s3dis/collect_indoor3d_data.py) & [/src/pointnet/preprocessing/s3dis/indoor3d_util.py](/src/pointnet/preprocessing/s3dis/indoor3d_util.py)
    1. Run preprocessing script

        ```
        cd PATH\generative-point-net\src\pointnet\preprocessing\s3dis
        python collect_indoor3d_data.py
        ```

    1. If message `ERROR! CHECK FILE NAME & PATH IF ERROR` uncomment code at **line 25** [/src/pointnet/preprocessing/s3dis/collect_indoor3d_data.py](/src/pointnet/preprocessing/s3dis/collect_indoor3d_data.py) to check output filename
    1. Processed data will save in ROOT_DIR/stanford_indoor3d in .npy format
    1. Example testing command

        ```
        UPCOMING
        ```
