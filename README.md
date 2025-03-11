# Generative Point Cloud Model

**Progress**

:white_check_mark: Complete dataset loaders for ModelNet, ShapeNetCore, S3DIS

:white_check_mark: PointNet & PointNet++ Classification Model Exploration

:white_check_mark: PointNet & PointNet++ Part Segmentation Model Exploration

:white_check_mark: PointNet & PointNet++ Segmentation Model Exploration

:white_large_square: Learn about Energy-Based Model (EBM)

:white_large_square: MCMC sampling with Langevin dynamics

:white_large_square: short-run Markov Chain Monte Carlo (MCMC)

:white_large_square: Generative PointNet Model Exploration

:white_large_square: New Generative Point Cloud Model

*pointnet and pointnet2 directories are used for learning and experimenting hence code is messy*

## Clone Latest

```
git clone --depth 1 https://github.com/luhouyang/generative-point-net.git
```

## Datasets

- **ModelNet40_Normal_Resampled** | [paper](https://arxiv.org/abs/1406.5670) | [dataset](https://www.kaggle.com/datasets/chenxaoyu/modelnet-normal-resampled) or [alternative source](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)

    1. Download the ModelNet40_Normal_Resampled dataset & unzip
    1. Use `--process_data` in all runs, first run will take longer to preproces and save data
    1. Example testing command

        ```
        cd PATH\generative-point-net
        python -m src.test.dataset_test modelnet40 PATH\ModelNet40 --process_data
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
        python -m src.test.dataset_test s3dis PATH\stanford_indoor3d
        ```

## Models

- **PointNet**

    1. Run classification training `--use_normal` flag to include normal in data

        ```
        cd PATH\generative-point-net
        python -m src.pointnet.main --output OUTPUT_DIR --dataset_path DATA_DIR --dataset modelnet10 --process_data
        ```

    1. Run segmentation training

        ```
        cd PATH\generative-point-net
        python -m src.pointnet.part_segmentation --output OUTPUT_DIR --dataset_path DATA_DIR --dataset shapenet
        ```

- **PointNet++**

    1. Run classification training `--use_normal` flag to include normal in data

        ```
        cd PATH\generative-point-net
        python -m src.pointnet2.train_cls --output OUTPUT_DIR --dataset_path DATA_DIR --dataset modelnet10 --process_data
        ```

    1. Run segmentation training

        ```
        cd PATH\generative-point-net
        python -m src.pointnet2.train_part_seg --output OUTPUT_DIR --dataset_path DATA_DIR --dataset shapenet
        ```

## Trained Models (No Normal)

- **PointNet**

    Classification

    1. [ModelNet10](https://drive.google.com/file/d/1vg9PlzLc-8lH8pGVFjkeTe-HQkHiPoZ-/view?usp=sharing)

    1. [ModelNet40](https://drive.google.com/file/d/1qKUTuDdPnP-rQ5ZZMmnUcAMtiPrnqs6u/view?usp=sharing)

    Part Segmentation

    1. [ShapeNetCore](https://drive.google.com/file/d/16JC6scMG_2xl2gw5zuED25k8aeCyXU-O/view?usp=sharing)

    Semantic Segmentation

    1. [S3DIS]()

- **PointNet++**

    Classification

    1. [ModelNet10](https://drive.google.com/file/d/1kDQYkjuE2uCgki_mF3nmnAodQOkli_vl/view?usp=sharing)

    1. [ModelNet40](https://drive.google.com/file/d/183Auoop7NOJvsTLi494ndBbirK58nvAC/view?usp=sharing)

    Part Segmentation

    1. [ShapeNetCore](https://drive.google.com/file/d/1xIasVRGiflhS_NVcsg7FIwmdrVg3qFxt/view?usp=sharing)

    Semantic Segmentation

    1. [S3DIS]()

## Classification Results

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

## Data Handlers

- **ModelNet**
    1. Example testing command (change between `modelnet40` or `modelnet10`)

        ```
        cd PATH\generative-point-net
        python -m src.test.datahandler_test modelnet40 PATH\ModelNet40
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
        cd PATH\generative-point-net
        python -m src.test.datahandler_test s3dis PATH\stanford_indoor3d
        ```

## References

**Datasets**

1. ModelNet - [paper](https://arxiv.org/abs/1406.5670) | [dataset](https://3dshapenets.cs.princeton.edu) or [alternative source](https://modelnet.cs.princeton.edu)

1. ShapeNetCore (subset) - [paper](https://dl.acm.org/doi/10.1145/2980179.2980238) | [dataset](https://www.kaggle.com/datasets/guxue17/shapenet1?select=shapenet)

1. Stanford Large-Scale Indoor Spaces 3D Dataset - [paper](https://ieeexplore.ieee.org/document/7780539) | [dataset](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform)

1. ShapeNet - [paper](https://arxiv.org/abs/1512.03012) | [dataset](https://shapenet.org/)

**Technical**

1. Generative PointNet - [paper](https://arxiv.org/abs/2004.01301) | [website](http://www.stat.ucla.edu/~jxie/GPointNet/) | [code](https://github.com/fei960922/GPointNet)

1. PointNet - [paper](https://arxiv.org/abs/1612.00593)

1. PointNet++ - [paper](https://arxiv.org/abs/1706.02413)

1. Generative Energy-Based Model - [paper](https://arxiv.org/abs/1602.03264) | [code & data](http://www.stat.ucla.edu/~ywu/GenerativeConvNet/main.html)

1. MCMC-Based Maximum Likelihood Learning of EBMs - [paper](https://arxiv.org/abs/1903.12370)

**Repository**

1. yanx27 - [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git)

1. fxia22 - [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch.git)

1. opeco17 - [pointnet](https://github.com/opeco17/pointnet.git)

## Results

- **PointNet**
    ### *Classification Training Loss & Accuracy ModelNet40*

    ![PointNet Training Loss](/src/pointnet/output/classification/no_normal/modelnet40/train_test_loss.png)

    ![PointNet Training Accuracy](/src/pointnet/output/classification/no_normal/modelnet40/train_test_acc.png)

    ### *Classification Prediction | 3/4 correct | 1/4 wrong*

    ![Prediction image 1](/archive/images/pointnet/prediction_1.png)

    ![Prediction image 2](/archive/images/pointnet/prediction_2.png)

    ### *Part Segmentation Training Loss & Accuracy*

    ![PointNet Training Loss](/src/pointnet/output/part_seg/train_test_loss.png)

    ![PointNet Training Accuracy](/src/pointnet/output/part_seg/train_test_acc.png)

- **PointNet++**
    ### *Classification Training Loss & Accuracy ModelNet40*

    ![PointNet Training Loss](/src/pointnet2/output/classification/msg/no_normal/modelnet40/train_test_loss.png)

    ![PointNet Training Accuracy](/src/pointnet2/output/classification/msg/no_normal/modelnet40/train_test_acc.png)

## Dataset Example Images

![ModelNet40 Example 3D Point Cloud](/archive/images/modelnet40_3d_pointcloud_image.png)

![ModelNet10 Example 3D Point Cloud](/archive/images/modelnet10_3d_pointcloud_image.png)

![ShapeNet Example 3D Point Cloud - Part Segmentation](/archive/images/shapenet_3d_pointcloud_part_segmentation_image.png)
