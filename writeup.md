[//]: # (Image References)
[image1]: ./assets/deconvnet.png
[image2]: ./assets/architecture.png

## Network Architecture
### Fully Convolutional Network
The task of Semantic Segmentation is a task of classifying each pixel in image. For the purpuse of the pixel classification, we can use usual convolution networks devised for image classification like AlexNet, VGG, ResNet, etc. By sliding the window over the entire image, the classification convolution networks can classify each window, and finally we can get a pixel classification map as a sementic segmentation of the image.

However, this approach of sliding window and classification by convnets is very computationary expensive. So, the network architecture called Fully Convolutional Network (FCN) is usually used in Semantic Segmentation.

![Encoder-Decoder Network][image1]
Figure 1 (from [Learning Deconvolution Network for Semantic Segmentation](http://cvlab.postech.ac.kr/research/deconvnet/))

The concept of FCN is visualized in Figure 1. It mainly has two parts: Encoder and Decoder.
</br>In the encoder, FCN extracts useful features by convolution and downsampling as convnets do for classification.
</br>Then, the last feature map of the encoder is passed to 1x1 convolution instead of fully connected layer (FC layer) usually used in classification convnets. 1x1 convolution is helpful to keep spatial information which is important for Semantic Segmentation, whereas FC layer loses the spatial information. Also, 1x1 convolution enables the model to take as input images of various sizes whereas FC layer only allows for fixed sized inputs.
</br>After the encoder and 1x1 convolution, the decoder takes the features encoded by the encoder. The decoder gradually upsamples features usually by non-learnable upsampling method or learnable transposed convolution, and finally produces a segmentation map for the image.

Skip connections between the encoder and decoder are usually helpful for the decoder to have spatially detailed information. By skip connection, each layer in the decoder takes as input the feature map of the corresponding layer in the encoder, which is usually more spatially detailed than upsampled output in the decoder, and the decoder can produce more spatially detailed, accurate segmentation.

### Architecture for the Follow Me project

The model I used in this problem has an architecture of U-Net, which can be thought as one of the FCN architectures.

In the encoder part of the architecture, it has 5 consecutive convolution layers with the doubling number of the filters. Each convolution layer spatially downsamples its input by stride = 2 instead of pooling. It is followed by Batch Normalization to have unit gaussian distributions of the activations of the convolution (, which is helpful for better training and some kind of regularization) and followed by ReLU to have non linearity.

After the encoder part, 1x1 convolution layer follows.

In the decoder part, the model has 5 consecutive layers of bilinear upsampling and covolution. Each decoding layer takes as input the output of the corresponding layer in the encoder part as well as the previous layer's output. By concatenating the two outputs, each decoding layer can preserve spatially detailed information, which is necessary to have accurate semantic segmentation map.

All of the convolution layers in the encoder and the decoder are separable convolution. Separable convolution convolve inputs separately in channelwise and spacewise. By separable convolution, we can reduce the computation of the convolution.

![Architecture][image2]

The model trained of the Follow Me dataset is specific to this problem (segmenting human or not), so we can't use the trained model for other problems like segmenting dogs, cats, cars, etc. However, the network architecture is not specific to this problem and can be adopted for other problems. We can use the network architecture and train models for other problems if we have enough datasets and annotations (i.e. segmentation masks).

## Hyper Parameters
**batch size**. I decided the batch size so that I could train the model on my GPU without out of memory errors, and I found that the batch size = 64 worked best.

**epoch**. I tried epoch = 2 but I found the model couldn't get enough training (underfitting). I also tried several longer epochs, but training for the longer epochs made the model overfit, and I couldn't get enough generalization performance. I finally found epoch = 5 is one of the best epochs that work fine for the problem without underfitting and overfitting.

**learning rate**. Having experiences of using Adam optimizer for some problems, I have an empirical knowledge that the learning rate = 0.001 often works best. So, I first tried the optimizer with the learning rate = 0.001. Because it worked fine, I didn't tune the learning rate further satisfied with the choise of the learning rate = 0.001.

**shift aug**. I didn't collect data other than the dataset provided because I wanted to try some data augmentations. I uncomment the [line 159-160](https://github.com/Fujiki-Nakamura/RoboND-DeepLearning-Project/blob/wip/code/utils/data_iterator.py#L159-L160) in the `code/utils/data_iterator.py` and used the provided `shift_and_pad_augmentation` ([line 65-85](https://github.com/Fujiki-Nakamura/RoboND-DeepLearning-Project/blob/wip/code/utils/data_iterator.py#L65-L85) in the script). The augmentation randomly shifts the images and masks by up to 15% of the size in the horizontal and vertical dimension. Without the data augmentation, I could only achieve   the final score of 0.3728, which is not enough for the minimum requirement of the problem (0.40).

## Reproduction
To reproduce the result with the trained weights, you can uncomment the cell `In [11]` in the notebook `code/model_training.ipynb`, restore the weights named `model_weights_8` and make predictions with the restored model. If you want to reproduce the result by training the model from scratch, you can leave the cell `In [11]` commented and run all the cells in the notebook with the hyperparameters defined in the cell `In [8]`.

## Future Enhancement
Although I didn't collect data from the simulator to have larger datasets, collecting additional data should help the model train better and improve the generalization performance of the model.
Of course, even with the larger dataset, we can take advantage of data augmentation. We can try some data augmentation other than shift augmentation such as gaussian noise, resizing and random erasing.

In addition to the data enhancement, we can enhance the network architecture. We might be able to use more advanced semantic segmentation architectures like SegNet, PSPNet, RefineNet, DeepLab, etc to get better performance.
