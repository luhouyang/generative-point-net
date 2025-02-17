# **Generative PointNet**

Learning and experimentation with [Generative PointNet](https://arxiv.org/abs/2004.01301)

## **Datasets**

- **ModelNet40** | [paper](https://arxiv.org/abs/1406.5670) | [dataset](https://3dshapenets.cs.princeton.edu) or [alternative source](https://modelnet.cs.princeton.edu)

    1. Download the ModelNet40 dataset & unzip
    1. Run preprocessing script [/src/pointnet/preprocessing/modelnet40/preprocessing.py](/src/pointnet/preprocessing/modelnet40/preprocess.py)
    1. Example testing command
    </br>

    ```
    cd PATH\generative-point-net
    python -m src.test.dataset_test modelnet40 PATH\ModelNet40 
    ```
    </br>

- **ModelNet10** | [paper](https://arxiv.org/abs/1406.5670) | [dataset](https://3dshapenets.cs.princeton.edu) or [alternative source](https://modelnet.cs.princeton.edu)

    1. Download the ModelNet10 dataset & unzip
    1. Delete `__MACOSX` directory
    1. Delete `raw` directory
    1. Delete all `.DS_Store` files
    1. Run preprocessing script [/src/pointnet/preprocessing/modelnet10/preprocessing.py](/src/pointnet/preprocessing/modelnet10/preprocess.py)
    1. Example testing command
    </br>

    ```
    cd PATH\generative-point-net
    python -m src.test.dataset_test modelnet10 PATH\ModelNet10 
    ```
    </br>

- **ShapeNet** | [paper](https://arxiv.org/abs/1512.03012) | [dataset](https://www.kaggle.com/datasets/guxue17/shapenet1?select=shapenet)

    1. Download the ShapeNet dataset & unzip
    1. Example testing command
    </br>

    ```
    cd PATH\generative-point-net
    python -m src.test.dataset_test shapenet PATH\shapenet\shapenetcore_partanno_segmentation_benchmark_v0 
    ```
    </br>

## **Models**

UPCOMING
