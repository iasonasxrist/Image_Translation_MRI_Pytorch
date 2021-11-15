# Image_Translation_MRI_Pytorch
 Medical image analysis using pytorch for image synthesis

In this project, we will be introducing you to the the role of a machine learning engineer/researcher at a healthcare technology company specializing in medical imaging applications. Our team wants to process and analyze magnetic resonance (MR) images of the brain. An MR imaging system is a flexible device that can create multiple types of images based on what a physician wants to see, but not all types of images are acquired in every scan due to time constraints. Your current processing and analysis algorithms require two types of MR images, but a new set of customer data only has one of those types. However, you have access to a fairly large, preprocessed dataset of paired examples of the two types of MR images, and you decide that deep learning would best perform this type of image transformation task.

You will use the deep learning framework PyTorch to implement a convolutional neural network for this task, and you will train it on the given paired data. We wants to make sure that the product is accurately performing the image translation, so you will need to provide qualitative and quantitative results demonstrating your method’s effectiveness.

Preprocessed subset of [IXI](https://brain-development.org/ixi-dataset/) MR brain images available [here](here).
However, you should have familiarity with the training and use of neural networks. The skills learned will be broadly useful for practitioners and researchers in deep learning; the specific techniques you will use are commonly used across computer vision tasks using deep learning, e.g., depth estimation in natural images with self-driving cars, super-resolution for any type of image, or removing rain from natural images. Additionally, you will gain familiarity with working with 3D data.

Definition:
Image-to-image translation is the process of transforming an image from one domain to another, where the goal is to learn the mapping between an input image and an output image.
![image](https://user-images.githubusercontent.com/80000902/141655130-dffb50ba-46f1-404b-a38a-ca32111b4a5e.png)
T1W vs T2W imaging.
Our goal is to transit from image to image form using Unet Deep Convolutional Neural Network.
 

UNET Architecture

Documentation:
https://arxiv.org/pdf/2010.02745v2.pdf

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8086713/
