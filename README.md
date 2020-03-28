# Semantic Labeling on Remote Sensing Corpora Using Feature Fusion-Based Enhanced Global Convolutional Network with High-Resolution Representations and Depthwise Atrous Convolution

Our previous work: [1] Panboonyuen, T.; Jitkajornwanich, K.; Lawawirojwong, S.; Srestasathiern, P.; Vateekul, P. Semantic Segmentation on Remotely Sensed Images Using an Enhanced Global Convolutional Network with Channel Attention and Domain Specific Transfer Learning. Remote Sens. 2019, 11, 83.
    Paper link: https://www.mdpi.com/2072-4292/11/1/83

GCN152-TL-A (our previous model) are developed based on the ResNet-152 backbone with Channel Attention and Domain Specific Transfer Learning:

<p align="center">
<img src="images/b1_method.png" width="800" />
</p>


Updates:

  - **Mar28: Fixed a few bugs and updated all checkpoints/results.**
  - **We will provide full codes after our latest article have been published.**

## 1. About HR-GCN-FF-DA Model (Our proposed deep learning architecture)

HR-GCN-FF-DA are a family of semantic segmentatiom models, which achieve state-of-the-art accuracy on Remote Sensing corpus, yet being an extracting features with higher quality and more efficient than our previous models.


HR-GCN-FF-DA are developed based on the advanced backbone, a new High-Resolution Representation (HR), and a modern deep learning technique:

<p align="center">
<img src="images/Graphical Abstract.png" width="800" />
</p>

  * **Backbone**: we employ the more advanced [HRNets](https://arxiv.org/abs/1908.07919) as our new backbone networks.
  * **DA**: we apply a new depthwise atrous convolution (DA) (from http://openaccess.thecvf.com/content_ICCV_2019/papers/Pang_Towards_Bridging_Semantic_Gap_to_Improve_Semantic_Segmentation_ICCV_2019_paper.pdf) feature network to enable easy and combine with our feature fusion. 


## 2. Pretrained HR-GCN-FF-DA Checkpoints

We have provided a list of EfficientNet checkpoints for HR-GCN-FF-DA checkpoints:.

  ** All checkpoints are trained with augmentation.
  Drive (Pretrained of HR-GCN-FF-DA for ISPRS Vaihingen): 
  Drive (Pretrained of HR-GCN-FF-DA for Landsat-8w3c corpus):
  Drive (Pretrained of HR-GCN-FF-DA for Landsat-8w5c corpus):


## 3. Run inference.

Files and Directories

- **train.py:** Training on the dataset of your choice. Default is Landsat-8w3c

- **test.py:** Testing on the dataset of your choice. Default is Landsat-8w3c

### Installation
This project has the following dependencies:

- TensorFlow `sudo pip install --upgrade tensorflow-gpu`

- OpenCV Python `sudo apt-get install python-opencv`

### Usage
The only thing you have to do to get started is set up the folders in the following structure:

    ├── "corpus_name"                   
    |   ├── train
    |   ├── train_labels
    |   ├── val
    |   ├── val_labels
    |   ├── test
    |   ├── test_labels

Put a text file under the dataset directory called "our_class_dict.csv" which contains the list of classes along with the R, G, B colour labels to visualize the segmentation results. This kind of dictionairy is usually supplied with the dataset. Here is an example for the Landsat-8w5c dataset:

```
name,r,g,b
Agriculture or Harvested area,64,128,64
Forest,192,0,128
Urban,0,128, 192
Water,0, 128, 64
Miscellaneous,128, 0, 0
```

## 4. Results.
**Note:** If you are using any of the networks that rely on a pre-trained HRNet or ResNet, then you will need to download the pre-trained weights using the provided script on section 2. These are currently: GCN152-TL-A and HR-GCN-FF-DA.

Then you can simply run `train.py`! Check out the optional command line arguments:

```
usage: train.py [-h] [--num_epochs NUM_EPOCHS]
                [--checkpoint_step CHECKPOINT_STEP]
                [--validation_step VALIDATION_STEP] [--image IMAGE]
                [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
                [--crop_height CROP_HEIGHT] [--crop_width CROP_WIDTH]
                [--batch_size BATCH_SIZE] [--num_val_images NUM_VAL_IMAGES]
                [--h_flip H_FLIP] [--v_flip V_FLIP] [--brightness BRIGHTNESS]
                [--rotation ROTATION] [--model MODEL] [--frontend FRONTEND]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --checkpoint_step CHECKPOINT_STEP
                        How often to save checkpoints (epochs)
  --validation_step VALIDATION_STEP
                        How often to perform validation (epochs)
  --image IMAGE         The image you want to predict on. Only valid in
                        "predict" mode.
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --crop_height CROP_HEIGHT
                        Height of cropped input image to network
  --crop_width CROP_WIDTH
                        Width of cropped input image to network
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --num_val_images NUM_VAL_IMAGES
                        The number of images to used for validations
  --h_flip H_FLIP       Whether to randomly flip the image horizontally for
                        data augmentation
  --v_flip V_FLIP       Whether to randomly flip the image vertically for data
                        augmentation
  --brightness BRIGHTNESS
                        Whether to randomly change the image brightness for
                        data augmentation. Specifies the max bightness change
                        as a factor between 0.0 and 1.0. For example, 0.1
                        represents a max brightness change of 10% (+-).
  --rotation ROTATION   Whether to randomly rotate the image for data
                        augmentation. Specifies the max rotation angle in
                        degrees.
  --model MODEL         The model you are using. See model_builder.py for
                        supported models
  --frontend FRONTEND   The frontend you are using. See frontend_builder.py
                        for supported models

```

## Results

These are some **sample results** for the Landsat-8w5c corpus with 5 classes 

<p align="center">
<img src="images/out1.png" width="800" />
</p>

These are some **sample results** for the Landsat-8w3c corpus with 3 classes 

<p align="center">
<img src="images/out3.png" width="800" />
</p>


These are some **sample results** for the ISPRS Vaihingen with 5 classes 

<p align="center">
<img src="images/out5.png" width="800" />
</p>

For more instructions about training on GPUs, please refer to the following tutorials:

  * Tensorflow tutorial: https://www.tensorflow.org/install/gpu

NOTE: this is still not an official code (untill we have published our article).

## Reference

[1] https://github.com/VXallset/deep-high-resolution-net.TensorFlow
[2] https://github.com/hsfzxjy/HRNet-tensorflow
[3] https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
[4] https://github.com/yoninachmany/raster-vision-deepglobe-semseg