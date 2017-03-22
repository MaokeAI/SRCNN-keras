# Keras implementation of SRCNN


The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)

<p align="center">
  <img src="https://github.com/MarkPrecursor/SRCNN-keras/blob/master/SRCNN.png" width="800"/>
</p>

My implementation have some difference with the original paper, include:

* use ['he_normal'](https://keras.io/initializations/) for weight initialization
* use Adam alghorithm for optimization, with learning rate 0.0003 for all layers.
* I use the opencv library to produce the training data 
* I did not set different learning rate in different layer, but I found this network still work.

## Use:
### Create your own data
open **prepare_data.py** and change the data path to your data

Excute:
`python prepare_data.py`

### training and test:
Excute:
`python main.py`


## Result(training for 200 epoches, with upscaling factor 2):
Origin Image:
<p align="center">
  <img src="https://github.com/MarkPrecursor/SRCNN-keras/blob/master/result.jpg" width="800"/>
</p>


