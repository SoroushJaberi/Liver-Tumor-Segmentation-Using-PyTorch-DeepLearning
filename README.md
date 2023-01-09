# Liver-Tumor-Segmentation-Using-PyTorch-DeepLearning-
First, we need to obtain and preprocess the data for the segmentation task The data is provided by the medical segmentation decathlon (http://medicaldecathlon.com/)
As this dataset has over 26GB we provide a resampled version of it. The new scans are of shape (256x256xZ), where Z is varying and reduce the size of the dataset to 2.5GB
You can directly download the full body cts and segmentation maps from:
https://drive.google.com/file/d/1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi/view?usp=sharing

# UNET
The idea behind a UNET is that we have "Downconvolutions" which are reducing the size of the image combined with increasing filter size followed by "Upconvolutions" which increase the image size up to the original size while reducing the number of filters.
All pairs between Up- and Downconvolutions are linked with skip connections.
Upsampling can either be done by interpolation or by UpConvolutions (ConvTranspose2d)

# Model Definition
We can use 2D-UNET architecture with some small changes:

Conv2d -> Conv3d
MaxPool2d -> MaxPool3d
"trilinear" upsampling method
Three Output Channels instead of One to model background, liver and tumor

# GPU: NVIDIA GeForce GTX 1050 Ti
