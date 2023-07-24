# LFSOD-CDINet
This project provides the code and results for 'Light Field Salient Object Detection with Sparse Views via Complementary and Discriminative Interaction Network', IEEE TCSVT, accepted, 2023. [paper link](https://ieeexplore.ieee.org/document/10168184)

The code will come in soon.

# Requirements
python 3.7 + TensorFlow 1.14.0

Note: The TensorFlow 1.x cannot initialize Layer_Norm with the pre-trained parameters, so we add a function 'layer_norm_initialized()' in the [layer.py](https://pan.baidu.com/s/18d5XrK3LlIGjbWWsCFLVog) (code: 7d8i).
You can put it under 'your_Anaconda_envs/Lib/site-packages/tensorflow/contrib/layers/python/layers/'.

# Saliency maps and performance

We provide [results](https://pan.baidu.com/s/1OSDsj9FCLZHMiTSGCPQ1Ww) (code: lau2) of our CDINet on 3 datasets (HFUT-Lytro Illum, HFUT-Lytro and DUTLF-V2)
<div align=center>
  <img src="https://github.com/GilbertRC/LFSOD-CDINet/blob/main/Images/CDINet.png">
</div>

# Training

# Test using pre-trained model

# Citation
```
@ARTICLE{Chen_2023_CDINet,
         author = {Yilei Chen and Gongyang Li and Ping An and Zhi Liu and Xinpeng Huang and Qiang Wu},
         title = {Light Field Salient Object Detection with Sparse Views via Complementary and Discriminative Interaction Network},
         journal = {IEEE Transactions on Circuits and Systems for Video Technology},
         year = {2023},
         doi = {10.1109/TCSVT.2023.3290600},
         }            
```
