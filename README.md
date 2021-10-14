[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/data-efficient-video-transformer-for-violence/action-recognition-on-real-life-violence)](https://paperswithcode.com/sota/action-recognition-on-real-life-violence?p=data-efficient-video-transformer-for-violence)


# Data-efficient-video-transformer

this repo is for menovideo associated with the paper ['Data Efficient Video Transformer for Violence Detection' (DeVTR)](https://ieeexplore.ieee.org/abstract/document/9530829)

one of big challenges facing researchers in computer vision with transformers especially in video tasks is the need for large data and high computational resources , our method called DeVTR (Data Efficient Video Transformer for Violence Detection) to overcame these challenges (he need for large data and high computational resources )

 In this work, we propose a data-efficient video transformer (DeVTr) based on the transformer network as a Spatio-temporal learning method with a pre-trained 2d-Convolutional neural network (2d-CNN) as an embedding layer for the input data. The model has been trained and tested on the Real-life violence dataset (RLVS) and achieved an accuracy of 96.25%. A comparison of the result for the suggested method with previous techniques illustrated that the suggested method provides the best result among all the other studies for violence event detection.

 ### Results and benchmarking
the model achieved 96.25% based on RLVS dataset and also worth to mention that it was better than TimeSformer in both memory efficiency and convergence speed and accuracy

[Comparing results of DeVTr vs other methods based on RLVS Dataset ](https://github.com/mamonraab/Data-efficient-video-transformer/blob/main/comptab.jpg)


[saliency map for random video of violence action ](https://github.com/mamonraab/Data-efficient-video-transformer/blob/main/fig5.jpg)


### menvideo package
the [menovideo package](https://pypi.org/project/menovideo/)   help you build video action recognation / video understanding  model  based on  
1-  build using our Novel model DeVTR with full costmaztion
2-  video dataset reader and preprocessing to easly read videos and make them as pytorch ready dataloaders
3-  Timedistributed warper similar to keras timedistributed warper which can help you easly build (classical CNN+LSTM )


this is new novel transformer network combined with Conv net to build a highly accuract video action recognation model with limited data and hw rescources 


### simple usage

install
```
pip install menovideo
 

```
import it
```
import menovideo.menovideo as menoformer
import menovideo.videopre as vide_reader 

```

init DeVTr model without pre-trained wights
```
model = menoformer.DeVTr()


```
init DeVTr with pre-trained wigths
the trained wights  can be [downloaded from this url](https://drive.google.com/file/d/1s7Z1c-4zC522BFVM5EiZDMQLe6ZV8QQh/view?usp=sharing)

```
wight = 'drive/MyDrive/Colab Notebooks/transformers/violance-detaction-myresearch/vg19bn40convtransformer-ep-0.pth'
model2 = menoformer.DeVTr(w= wight , base ='default')

```


using the video reader and pre-processing helpers
parameters is :

1.  pandas dataframe contain the path and label of each video
2.  number of frames for the singal video
3.  RGB is the number of color channles
4.  h is the hieght of the frame for each video
5.  w is the width of the frame for each video
```
valid_dataset = vide_reader.TaskDataset(valid_df,timesep=time_stp,rgb=RGB,h=H,w=W)

```

for detlied example of using the labrary use [package_test.ipynb](https://github.com/mamonraab/Data-efficient-video-transformer/blob/main/package_test.ipynb)

#### please use pytorch 1.9 for the pre-trained model 

To cite our paper/code:

```

@INPROCEEDINGS{9530829,  author={Abdali, Almamon Rasool},  booktitle={2021 IEEE International Conference on Communication, Networks and Satellite (COMNETSAT)},   title={Data Efficient Video Transformer for Violence Detection},   year={2021},  volume={},  number={},  pages={195-199},  doi={10.1109/COMNETSAT53002.2021.9530829}}

```
