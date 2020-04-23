# PigSkin-CNN

<p>
  For this project we use the <a href = 'https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000'>"Skin Cancer MNIST: HAM10000"</a> database, which consists of a set of 10,015 dermoscopic images collected by different doctors. Lesions are described in https://www.nature.com/articles/sdata2018161.pdf. The size of the original RGB images is 450x600 pixels and they were adjusted to a size of 32x32 for network training.
</p>

In this project, the training of a supervised learning algorithm for the classification of skin lesions was carried out. Some categories had few images as shown in the following figure:

<p align='center'>
  <img src= 'https://github.com/Luisbaduy97/PigSkin-CNN/blob/master/histo_original.png'>
</p>


therefore, the implementation of algorithms for the rotation of some images had to be carried out as shown in the following figure.




<p align='center'>
  <img src= 'https://github.com/Luisbaduy97/PigSkin-CNN/blob/master/rotaciones.png'>
</p>



After data augmentation, it can be seen how the categories increase significantly.

<p align='center'>
  <img src= 'https://github.com/Luisbaduy97/PigSkin-CNN/blob/master/aumented_histogram.png'>
</p>


<p>
  The implementation of a basic architecture proposed by <a href = 'http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf'>Yann LeCun</a>, the architecture is shown in the following figure:
</p>

<p align='center'>
  <img src= 'https://github.com/Luisbaduy97/PigSkin-CNN/blob/master/arquitectura.png'>
</p>

After the network training it was validated with a set of 9912 and an accuracy of 85% was obtained, the weights were saved in a file and uploaded to a python file to develop a system capable of identifying new images from the hyperparameters obtained by the convolutional nueronal network.

