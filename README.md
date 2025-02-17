# **Generative PointNet**

Learning and experimentation with [Generative PointNet](https://arxiv.org/abs/2004.01301)

## **Datasets**

- **ModelNet40** | [paper](https://arxiv.org/abs/1406.5670) | [dataset](https://3dshapenets.cs.princeton.edu) or [alternative source](https://modelnet.cs.princeton.edu)

    1. Download the ModelNet40 dataset & unzip
    1. Copy the classes file [/src/pointnet/preprocessing/modelnet40/train.txt](/src/pointnet/preprocessing/modelnet40/train.txt) into the root folder `PATH/ModelNet40/train.txt`
    1. Run preprocessing script [/src/pointnet/preprocessing/modelnet40/preprocessing.py](/src/pointnet/preprocessing/modelnet40/preprocess.py)
    1. Uncomment code at bottom to test dataset loading [/src/pointnet/dataset.py](/src/pointnet/dataset.py)
    1. Example command
    </br>
    
    ```
    cd PATH\generative-point-net\src\pointnet
    python dataset.py modelnet40 PATH\ModelNet40 
    ```

- **ModelNet10** | [paper](https://arxiv.org/abs/1406.5670) | [dataset](https://3dshapenets.cs.princeton.edu) or [alternative source](https://modelnet.cs.princeton.edu)

    1. Download the ModelNet10 dataset & unzip
    1. Delete `__MACOSX` directory
    1. Delete `raw` directory
    1. Delete all `.DS_Store` files
    1. Copy the classes file [/src/pointnet/preprocessing/modelnet10/train.txt](/src/pointnet/preprocessing/modelnet10/train.txt) into the root folder `PATH/ModelNet10/train.txt`
    1. Run preprocessing script [/src/pointnet/preprocessing/modelnet10/preprocessing.py](/src/pointnet/preprocessing/modelnet10/preprocess.py)
    1. Uncomment code at bottom to test dataset loading [/src/pointnet/dataset.py](/src/pointnet/dataset.py)
    1. Example command
    </br>

    ```
    cd PATH\generative-point-net\src\pointnet
    python dataset.py modelnet10 PATH\ModelNet10 
    ```

- **ShapeNet** | [paper](https://arxiv.org/abs/1512.03012) | [dataset](https://www.kaggle.com/datasets/guxue17/shapenet1?select=shapenet)

    1. Download the ShapeNet dataset & unzip
    1. Uncomment code at bottom to test dataset loading [/src/pointnet/dataset.py](/src/pointnet/dataset.py)
    1. Example command
    </br>

    ```
    cd PATH\generative-point-net\src\pointnet
    python dataset.py shapenet PATH\shapenet\shapenetcore_partanno_segmentation_benchmark_v0 
    ```

## **Models**

UPCOMING
