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

- **ShapeNetCore** | [paper](https://arxiv.org/abs/1512.03012) | [dataset](https://www.kaggle.com/datasets/guxue17/shapenet1?select=shapenet)

    1. Download the ShapeNetCore dataset & unzip
    1. Example testing command
    </br>

    ```
    cd PATH\generative-point-net
    python -m src.test.dataset_test shapenet PATH\shapenet\shapenetcore_partanno_segmentation_benchmark_v0 
    ```
    </br>

- **Stanford Large-Scale Indoor Spaces 3D Dataset (S3DIS)** | [paper](https://ieeexplore.ieee.org/document/7780539) | [dataset](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform)

    1. UPCOMING
    </br>

## **Models**

UPCOMING

## **Data Handlers**

- **ModelNet40**
    ```
    cd PATH\generative-point-net
    python -m src.test.datahandler_test modelnet40 PATH\ModelNet40
    ```

- **ModelNet10**
    ```
    cd PATH\generative-point-net
    python -m src.test.datahandler_test modelnet10 PATH\ModelNet10
    ```

- **ShapeNetCore**
    ```
    cd PATH\generative-point-net
    python -m src.test.datahandler_test shapenet PATH\shapenet\shapenetcore_partanno_segmentation_benchmark_v0
    ```

- **Stanford Large-Scale Indoor Spaces 3D Dataset (S3DIS)**
    ```
    UPCOMING
    ```

## **References**

**Datasets**

1. ModelNet - [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://arxiv.org/abs/1406.5670) | [dataset](https://3dshapenets.cs.princeton.edu) or [alternative source](https://modelnet.cs.princeton.edu)

1. ShapeNetCore (subset) - [A Scalable Active Framework for Region Annotation in 3D Shape Collections](https://dl.acm.org/doi/10.1145/2980179.2980238) | [dataset](https://www.kaggle.com/datasets/guxue17/shapenet1?select=shapenet)

1. Stanford Large-Scale Indoor Spaces 3D Dataset - [3D Semantic Parsing of Large-Scale Indoor Spaces](https://ieeexplore.ieee.org/document/7780539) | [dataset](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform)

1. ShapeNet - [ShapeNet: An Information-Rich 3D Model Repository](https://arxiv.org/abs/1512.03012) | [dataset](https://shapenet.org/)

**Technical**

1. Generative PointNet - [Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction and Classification](https://arxiv.org/abs/2004.01301) | [website](http://www.stat.ucla.edu/~jxie/GPointNet/) | [code](https://github.com/fei960922/GPointNet)

1. PointNet - [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)

1. Generative Energy-Based Model - [A Theory of Generative ConvNet](https://arxiv.org/abs/1602.03264) | [code & data](http://www.stat.ucla.edu/~ywu/GenerativeConvNet/main.html)

1. MCMC-Based Maximum Likelihood Learning of EBMs - [On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models](https://arxiv.org/abs/1903.12370) 

## **Dataset Example Images**

![ModelNet40 Example 3D Point Cloud](/archive/images/modelnet40_3d_pointcloud_image.png)

![ModelNet10 Example 3D Point Cloud](/archive/images/modelnet10_3d_pointcloud_image.png)

![ShapeNet Example 3D Point Cloud - Part Segmentation](/archive/images/shapenet_3d_pointcloud_part_segmentation_image.png)