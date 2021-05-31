
<h1 align="center">Face Mask Detection</h1>

<div align= "center"><img src="https://github.com/techyhoney/Facemask_Detection/blob/main/facemasklogo1.png"/>
  
<h4>Maskd (Face Mask Detection system) built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts in order to detect face masks in static images as well as in real-time video streams.</h4>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
<img src = "https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white"/>
<img src ="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white"/>

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/chandrikadeb7/Face-Mask-Detection/issues)
[![Website monip.org](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](http://monip.org/)
[![GitHub contributors](https://img.shields.io/github/contributors/Naereen/badges.svg)](https://GitHub.com/Naereen/badges/graphs/contributors/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Live Demo](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/Readme_images/Demo.gif)


## :muscle: Motivation
In the present scenario due to Covid-19, there is no efficient face mask detection applications which are now in high demand for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety. Also, the absence of large datasets of __‘with_mask’__ images has made this task more cumbersome and challenging. 

 
## 💻 Live Demo
See how the web app looks 👉 @ hitengoyal.me

[![Already deployed version](https://raw.githubusercontent.com/vasantvohra/TrashNet/master/hr.svg)](https://face-mask--detection-app.herokuapp.com/)


## :warning: TechStack/framework used

- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Streamlit](https://docs.streamlit.io/en/stable/api.html)

## :star: Usecase
Our face mask detector didn't use any morphed masked images dataset. The model is accurate, and since we used the CNN architecture, it’s also computationally efficient and thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.).

This system can therefore be used in real-time applications which require face-mask detection for safety purposes due to the outbreak of Covid-19. This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.

## :file_folder: Dataset
The dataset used can be downloaded here - [Click to Download](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)

This dataset consists of __4000 images__ belonging to two classes:
*	__with_mask: 2000 images__
*	__without_mask: 2000 images__

The images used were real images of faces wearing masks. The images were collected from the following sources:

* [__Kaggle datasets__](https://www.kaggle.com/search?q=facemask+detection+in%3Adatasets)
* [__RMFD dataset__](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)
* [__Google Dataset Search__](https://datasetsearch.research.google.com/)

## :gear: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/requirements.txt)

## 🚀&nbsp; Installation
1. Clone the repo
```
$ git clone https://github.com/chandrikadeb7/Face-Mask-Detection.git
```

2. Change your directory to the cloned repo 
```
$ cd Face-Mask-Detection
```

3. Create a Python virtual environment named 'test' and activate it
```
$ virtualenv test
```
```
$ source test/bin/activate
```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

## :bulb: Working

1. Open terminal. Go into the cloned project directory and type the following command:
```
$ python3 train_mask_detector.py --dataset dataset
```

2. To detect face masks in an image type the following command: 
```
$ python3 detect_mask_image.py --image images/pic1.jpeg
```

3. To detect face masks in real-time video streams type the following command:
```
$ python3 detect_mask_video.py 
```
## :trophy: Results

#### Our model gave around 99% accuracy for Face Mask Detection after training.
####          
![](https://i.imgur.com/3vo1w8f.png)

####          

#### We got the following accuracy/loss training curve plots
![](https://i.imgur.com/cLNo6nK.png)
####          
![](https://i.imgur.com/RYiOlCP.png)


## Streamlit app

Face Mask Detector webapp using Tensorflow & Streamlit command
```
$ streamlit run deploy.py 
```
## :handshake: Contribution
Feel free to **file a new issue** with a respective title and description on the the [Facemask_Detection](https://github.com/techyhoney/Facemask_Detection/issues) repository. If you already found a solution to your problem, **We would love to review your pull request**! 


## :handshake: Our Contributors

[CONTRIBUTORS.md](/CONTRIBUTORS.md)

## :eyes: License
[MIT](https://github.com/techyhoney/Facemask_Detection/blob/main/LICENSE)

