# Smooth Grad-CAM++ with pytorch
The re-implementation of Smooth Grad-CAM++ with pytorch.
This repo also includes the code of CAM, Grad-CAM and Grad-CAM++.

## Requirements
* python 3.x
* pytorch >= 0.4
* pillow
* numpy
* opencv
* matplotlib

## How to use
You can use the CAM, GradCAM, GradCAM++ and Smooth Grad-CAM++ as a model wrapper described in `cam.py`.
Please see `demo.ipynb` for the detail.

# Results
|image|![](results/tigercat.jpg)|![](samples/dogsled.jpg)|
|:-:|:-:|:-:|
|CAM|![](tiger cat_cam.png)|![](samples/dogsled, dog sled, dog sleigh_cam.png)|
|:-:|:-:|:-:|
|Grad-CAM|![](tiger cat_gradcam.png)|![](samples/dogsled, dog sled, dog sleigh_gradcam.png)|
|:-:|:-:|:-:|
|Grad-CAM++|![](tiger cat_gradcampp.png)|![](samples/dogsled, dog sled, dog sleigh_gradcampp.png)|
|:-:|:-:|:-:|
|Smooth Grad-CAM++|![](tiger cat_smoothgradcampp.png)|![](samples/dogsled, dog sled, dog sleigh_smoothgradcampp.png)|

## References
* Smooth Grad-CAM++: An Enhanced Inference Level Visualization Technique for Deep Convolutional Neural Network Models,  
  Daniel Omeiza, Skyler Speakman, Celia Cintas, Komminist Weldermariam [[paper](https://arxiv.org/abs/1908.01224)]
* Learning Deep Features for Discriminative Localization, 
  Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba [[paper](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)]
* Grad-CAM: Visual explanations from deep networks via gradient-based localization,
  Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra, [[arXiv](https://arxiv.org/abs/1610.02391)]
* Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks,
  Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader and Vineeth N Balasubramanian[[arXiv](https://arxiv.org/pdf/1710.11063.pdf)]
  
