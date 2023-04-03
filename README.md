# Scanpath Feature Engineering for Image Recognition Models

This repository provides a way to visualize scanpath by incorporating different elements like temporal information, fixation duration, aggregation of fixations on pre-defined AOIs (areas of interest), etc. 

We refined the code used [here](https://assets.researchsquare.com/files/rs-2088288/v1_covered.pdf?c=1667195112) which is a modified version of [PyGaze](https://link.springer.com/article/10.3758/s13428-013-0422-2) for better modularization of each scanpath feature. We hope any prospective researcher looking to replicate the same visualization strategies in
this paper may find this implementation easily generalizable to their needs. The module GazePlotter contains a class of the same name which saves different scanpath features such as fixations, saccades, raw gaze data, and AOI information and allows for the user to generate images by calling individual functions (eg. calling draw saccades draws the saccades on a Matplotlib figure saved as an instance attribute which can later be saved using the function save fig ). This can further be automated by using the function run pipeline which accepts a Python dictionary containing the name of each function the user wants to call as well as its corresponding parameters (if any). The image configuration template configs/image cgs/image config template.yml provides a complete example of a GazePlotter pipeline. The image below shows a simplified example of how GazePlotter can be run using a pipeline. A concrete demonstration of how GazePlotter is used is found in ImagePrep.py.

![gazeplotterPipeline](https://user-images.githubusercontent.com/93376103/226864153-aa3aba21-c76b-4641-bf19-6d8bea5fc59a.PNG)


# Models for Evaluation 
To compare the effectiveness of each configuration, we evaluated scanpath sets generated from various settings on two standard image classifiers using 5-fold cross validation for both models as well as an additional standard 70-20-10 train-validate-test split on the VGG-16 model. 
## VGG-16
We used a standard off-the-shelf VGG-16 model pretrained on the [Imagenet](https://arxiv.org/abs/1409.1556) dataset. We chose this model due to both its popularity and ease to implement. The model was created using the Pytorch library, which makes the Imagenet weights available for VGG models as well as many other standard architecture. 

## SVM 
We also compared the results of the VGG-16 model to a Support Vector Machine (SVM) image classifier using the ![SKlearn](https://scikit-learn.org/stable/) library. For each scanpath set, we apply an exhaustive parameter grid search to select the values of the hyperparameters. The models fit with the best parameters per scanpath set are available for download ![here](https://osf.io/fhmjy/). 

# Dataset 
We produced 13 scanpath sets from the dataset in ![Marchiori's paper](https://www.sciencedirect.com/science/article/abs/pii/S0022053121001083). The data was recorded during a behavioral experiment where the eye movements of participants are captured while playing a set of economic games. The generated scanpath images are available for download ![here](https://osf.io/fhmjy/). 
