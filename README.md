# LFSOD-CDINet
This project provides the code and results for 'Light Field Salient Object Detection with Sparse Views via Complementary and Discriminative Interaction Network', IEEE TCSVT, 2023. [paper link](https://ieeexplore.ieee.org/document/10168184)

## Network architecture
<div align=center>
<img src="https://github.com/GilbertRC/LFSOD-CDINet/blob/main/Figs/Network.png">
</div>

## Requirements
python 3.7 + TensorFlow 1.14.0

Note: We provide a modified [layer.py](https://pan.baidu.com/s/18d5XrK3LlIGjbWWsCFLVog) (code: 7d8i) for TensorFlow 1.14.0. The added `layer_norm_initialized()` enables initializing Layer_Norm with pre-trained parameters.
You can put it under 'your_Anaconda_envs/Lib/site-packages/tensorflow/contrib/layers/python/layers/' folder.

## Training
1. Download the [TrainingSet](https://pan.baidu.com/s/1HNWVOFEkIOPUz3u2s3AhCQ) (code: t7gt) and put it under './dataset/' folder.
2. Download the pre-trained [vgg-16 model](https://pan.baidu.com/s/1ZJKXk2zR-Mv8Aq5YYifm0g) (code: kq1o) and [mpi model](https://pan.baidu.com/s/1eGziqgmrC9VGQpHasEW4IA) (code: c3tj) and put them under './models/' folder.
3. Run `train.py` (default to the HFUT-Lytro Illum dataset).

## Test using pre-trained model
1. Download the [TestSet](https://pan.baidu.com/s/17FNkxtXYBTtLJI8s5xy_gw) (code: hdl2) and put it under './dataset/' folder.
2. Download our pre-trained [model_HFUT](https://pan.baidu.com/s/11lqmaCoatJ4K-GquW1izyA) (code: k28i) and [model_DUTLF-V2](https://pan.baidu.com/s/1TKeAhc1GYHTGGc7bdwPL8w) (code: h8ou) and put them under './checkpoints/' folder. 
3. Run `test.py`. The SOD results will be saved under './results/' folder.

Note: In the paper, we use model_HFUT to test the HFUT-Lytro Illum & HFUT-Lytro datasets and use model_DUTLF-V2 to test the DUTLF-V2 dataset.

## Saliency maps and performance
We provide [results](https://pan.baidu.com/s/1OSDsj9FCLZHMiTSGCPQ1Ww) (code: lau2) of our CDINet on 3 datasets (HFUT-Lytro Illum, HFUT-Lytro and DUTLF-V2)
<div align=center>
  <img src="https://github.com/GilbertRC/LFSOD-CDINet/blob/main/Figs/Table.png">
</div>

## Citation
```
@ARTICLE{LFSOD-CDINet,  
  title={Light Field Salient Object Detection with Sparse Views via Complementary and Discriminative Interaction Network},
  author={Yilei Chen and Gongyang Li and Ping An and Zhi Liu and Xinpeng Huang and Qiang Wu},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  volume={34},
  number={2},
  pages={1070-1085},
  month={Feb.}}            
```

Any questions regarding this work can contact yileichen@shu.edu.cn.
