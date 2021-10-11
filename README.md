# Data-efficient-video-transformer

this repo is for menovideo associated with the paper ['Data Efficient Video Transformer for Violence Detection' (DeVTR)](https://ieeexplore.ieee.org/abstract/document/9530829)
the meno packge   help you build video action recognation model  based on  our Novel model DeVTR 

this is new novel transformer network combined with Conv net to build a highly accuract video action recognation model with limted data and hw rescources 

 In this work, we propose a data-efficient video transformer (DeVTr) based on the transformer network as a Spatio-temporal learning method with a pre-trained 2d-Convolutional neural network (2d-CNN) as an embedding layer for the input data. The model has been trained and tested on the Real-life violence dataset (RLVS) and achieved an accuracy of 96.25%. A comparison of the result for the suggested method with previous techniques illustrated that the suggested method provides the best result among all the other studies for violence event detection.

the trained wights  can be downloaded from this url  https://drive.google.com/file/d/1s7Z1c-4zC522BFVM5EiZDMQLe6ZV8QQh/view?usp=sharing



## please use pytorch 1.9 for the pre-trained model 

for detlied example of using the labrary use [package_test.ipynb](https://github.com/mamonraab/Data-efficient-video-transformer/blob/main/package_test.ipynb)


To cite our paper/code:

```

@INPROCEEDINGS{9530829,  author={Abdali, Almamon Rasool},  booktitle={2021 IEEE International Conference on Communication, Networks and Satellite (COMNETSAT)},   title={Data Efficient Video Transformer for Violence Detection},   year={2021},  volume={},  number={},  pages={195-199},  doi={10.1109/COMNETSAT53002.2021.9530829}}

```
