# **Generative Point Cloud Model**

**Progress**

:white_check_mark: Complete dataset loaders for ModelNet, ShapeNetCore, S3DIS

:white_check_mark: PointNet Classification & Part Segmentation Model Exploration

:white_large_square: PointNet++ Classification & Segmentation Model Exploration

:white_large_square: Learn about Energy-Based Model (EBM)

:white_large_square: MCMC sampling with Langevin dynamics

:white_large_square: short-run Markov Chain Monte Carlo (MCMC)

:white_large_square: Generative PointNet Model Exploration

:white_large_square: New Generative Point Cloud Model

## **Clone Latest**

```
git clone --depth 1 https://github.com/luhouyang/generative-point-net.git
```

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

## **Models**

- **PointNet**

    1. Run classification training

        ```
        cd PATH\generative-point-net
        python -m src.pointnet.main --output OUTPUT_DIR --dataset_path DATA_DIR --dataset shapenet
        ```

    1. Run segmentation training

        ```
        cd PATH\generative-point-net
        python -m src.pointnet.part_segmentation --output OUTPUT_DIR --dataset_path DATA_DIR --dataset shapenet
        ```

## **Classification Results**

- **PointNet**

    *please check results section for more information on training*

    1. To run test, first download the [ShapeNetCore Classification trained model](https://drive.google.com/file/d/10bx_57_JCfq6G9Ql1hnd_GpZtPzXhdOm/view?usp=sharing)
    
        ```
        cd PATH\generative-point-net
        python -m src.pointnet.visualize --model_path MODEL_PATH.pth --dataset_path DATA_DIR --num_samples 4
        ```

    1. To run test, first download the [ShapeNetCore Part Segmentation trained model](https://drive.google.com/file/d/1RsDBG_priDfwKhZPI-ug8hm7yOf6yXLq/view?usp=sharing)

        ```
        cd PATH\generative-point-net
        python -m src.pointnet.visualize_part_seg --model_path MODEL_PATH.pth --dataset_path DATA_DIR --num_samples 4
        ```

## **Data Handlers**

- **ModelNet40**
    1. Example testing command

        ```
        cd PATH\generative-point-net
        python -m src.test.datahandler_test modelnet40 PATH\ModelNet40
        ```

- **ModelNet10**
    1. Example testing command
    
        ```
        cd PATH\generative-point-net
        python -m src.test.datahandler_test modelnet10 PATH\ModelNet10
        ```

- **ShapeNetCore**
    1. Example testing command
    
        ```
        cd PATH\generative-point-net
        python -m src.test.datahandler_test shapenet PATH\shapenet\shapenetcore_partanno_segmentation_benchmark_v0_normal
        ```

- **Stanford Large-Scale Indoor Spaces 3D Dataset (S3DIS)**
    1. Example testing command

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

1. PointNet++ - [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)

1. Generative Energy-Based Model - [A Theory of Generative ConvNet](https://arxiv.org/abs/1602.03264) | [code & data](http://www.stat.ucla.edu/~ywu/GenerativeConvNet/main.html)

1. MCMC-Based Maximum Likelihood Learning of EBMs - [On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models](https://arxiv.org/abs/1903.12370)

**Repository**

1. yanx27 - [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git)

1. fxia22 - [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch.git)

1. opeco17 - [pointnet](https://github.com/opeco17/pointnet.git)

## **Results**

- **PointNet**
    ### *Classification Training Loss & Accuracy*

    ![PointNet Training Loss](/src/pointnet/output/classification/shapenetcore/train_test_loss.webp)

    ![PointNet Training Accuracy](/src/pointnet/output/classification/shapenetcore/train_test_acc.webp)

    ### *Classification Prediction | 3/4 correct | 1/4 wrong*

    ![Prediction image 1](/archive/images/pointnet/prediction_1.png)

    ![Prediction image 2](/archive/images/pointnet/prediction_2.png)

    ### *Part Segmentation Training Loss & Accuracy*

    ![PointNet Training Loss](/src/pointnet/output/part_seg/train_test_loss.png)

    ![PointNet Training Accuracy](/src/pointnet/output/part_seg/train_test_acc.png)

## **Trained Models**

- **PointNet**

    Classification

    1. [ModelNet10](https://drive.google.com/file/d/16z72KgrnAcAB9U4zjA95E147uZFb2zjD/view?usp=sharing)

    1. [ModelNet40](https://drive.google.com/file/d/1NwUDitpEFFnwJoQ9RAqlfNKFcxOcC5s8/view?usp=sharing)

    1. [ShapeNetCore](https://drive.google.com/file/d/10bx_57_JCfq6G9Ql1hnd_GpZtPzXhdOm/view?usp=sharing)

    Part Segmentation

    1. [ShapeNetCore]()

    Semantic Segmentation

    1. [S3DIS]()

- **PointNet++**

    Classification

    1. [ModelNet10]()

    1. [ModelNet40]()

    1. [ShapeNetCore]()

    Part Segmentation

    1. [ShapeNetCore]()

    Semantic Segmentation

    1. [S3DIS]()

## **Dataset Example Images**

![ModelNet40 Example 3D Point Cloud](/archive/images/modelnet40_3d_pointcloud_image.png)

![ModelNet10 Example 3D Point Cloud](/archive/images/modelnet10_3d_pointcloud_image.png)

![ShapeNet Example 3D Point Cloud - Part Segmentation](/archive/images/shapenet_3d_pointcloud_part_segmentation_image.png)
