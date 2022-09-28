# MoNuSeg 2018 challenge (Submission)
Contains an implementation of a UNet and a HRNet model along with attention gated networks applied to them. The final accuracy is best for the HRNet + attention gated module.

## Content
Consists of implementations of some semantic segmentation models. Namely, [UNet](https://arxiv.org/pdf/1505.04597.pdf) , [HRNet for semantic segmentation](https://arxiv.org/abs/1908.07919) and [Mask-RCNN](https://arxiv.org/abs/1703.06870). The implementations are in their respective notebooks.

### Mask-RCNN
The implementation is present [here](https://github.com/advaitkumar3107/MoNuSeg/blob/master/mask_rcnn/mask_rcnn.ipynb). I was planning to use this initially for my solution, but it required a lot of memory to train. Hence I could train it properly only for a few epochs. I had trained it for about 20 epochs(10 for fine tuning and 10 for training heads) and the results were not that good. A sample result is included in the colab notebook(although for instance segmentation). As can be seen, it isnt that great.

### UNet + HRNet + Gated Attention Networks
The problems in Mask-RCNN caused me to shift to a simpler model, namely UNet[(implementation)](https://github.com/advaitkumar3107/MoNuSeg/blob/master/solution/unet_segmentation.ipynb) and HRNet[(implementation)](https://github.com/advaitkumar3107/MoNuSeg/blob/master/solution/HRNet_segmentation.ipynb). 
The architecture of UNet is described below:

<p align='center'>  
  <img src='https://github.com/advaitkumar3107/MoNuSeg/blob/master/Images/unet_architecture.png' width='870'/>
</p>

The architecture of HRNet is described below:

<p align='center'>  
  <img src='https://github.com/advaitkumar3107/MoNuSeg/blob/master/Images/hrnet_architecture.png' width='870'/>
</p>
The outputs from all the 4 branches are concatenated together and convolved into 1 channel for the output prediction.


The architecture of the gated attention network is described below:

<p align='center'>  
  <img src='https://github.com/advaitkumar3107/MoNuSeg/blob/master/Images/attention.png' width='870'/>
</p>
The gate channel contains feature maps from the coarser regions of the network and the input channel contain the original input feature maps.

The dataset was split into [training](https://github.com/advaitkumar3107/MoNuSeg/tree/master/Datasets/nucleus) and [testing](https://github.com/advaitkumar3107/MoNuSeg/tree/master/Datasets/test). While training, the training dataset was randomly split into training images(80%) and validation images(20%). For pre-processing, the data was randomly augmented(flipped, mirrored, translated). Each 1000X1000 image was split into 25, 200X200 patches for training.

### Results
F1 score = 0.7591 <br />
iou = 0.6135

I also used the jaccard_score metric from sklearn.metrics, to calculate accuracy. The code for testing the model and displaying results is present [here](https://github.com/advaitkumar3107/MoNuSeg/blob/master/solution/Monuseg_Prediction.ipynb). The accuracy is calculated on the unseen 14 test images.

Accuracy of the final HRNet + gated attention model = 0.6731

### Visualization
The final outputs for visualizing were constructed by joining the 25, 200X200 patches in the same fashion as the original image was split. 
Here are the results from the UNet Model with attention:
<p align='center'>  
  <img src='https://github.com/advaitkumar3107/MoNuSeg/blob/master/Images/unet_predictions.png' width='870'/>
</p>

Image on the left is input, the centre one is the prediction and the image on the right is the actual mask.

Results from the HRNet Model with attention:
<p align='center'>  
  <img src='https://github.com/advaitkumar3107/MoNuSeg/blob/master/Images/hrnet_predictions.png' width='870'/>
</p>

The predictions for all the 14 test images are present [here](https://github.com/advaitkumar3107/MoNuSeg/tree/master/Datasets/prediction)
