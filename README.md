SketchGAN is a deep learning project focused on generating realistic images from sketches using a Generative Adversarial Network (GAN) model. The project aims to assist in criminal identification by creating potential criminal likenesses based on witness or police sketches. It employs Conditional GANs (cGANs) to produce high-quality images that are conditioned on the input sketches. This project was completed in collaboration with Pratheesh (lnu.prat@northeastern.edu).

The project includes modules for training, testing, hyperparameter tuning, and evaluating the performance of different models. In addition, the project allows for the comparison of models based on various evaluation metrics such as the Fréchet Inception Distance (FID) and Inception Score (IS).

Features:
Sketch-to-Image Generation: Generates realistic images from hand-drawn or synthetic sketches.
Conditional GAN (cGAN) Architecture: Utilizes GANs conditioned on input sketches for image generation.
Criminal Identification: Applies generated images to potential criminal identification scenarios, allowing comparison of generated images to existing criminal databases.
Model Evaluation and Hyperparameter Tuning: Includes evaluation scripts and a grid search mechanism for selecting the best model configuration.
Dataset Loader and Preprocessing: Efficiently loads and preprocesses image datasets.

Project Structure
The following files are integral to the project:

dataset_loader.py: Script responsible for loading and preprocessing the dataset of sketches and real images.

train.py: Main script for training the GAN model. This script sets up the generator and discriminator, defines the loss function and optimizers, and logs the training progress.

test.py: Script for testing the model with unseen sketches and generating corresponding images.

evaluate.py: Used for evaluating the trained model using metrics such as FID and IS.

evaluate_models.py: Script for comparing different trained models.

gridsearch.py: Script for performing grid search on hyperparameters to optimize the GAN model.

models/: Contains model definitions for the generator and discriminator.

model_evaluation_results.json: Stores results from model evaluations, including metrics such as FID and IS for multiple models.

Data/: Contains the dataset of sketches and real images.

Model Download Instructions
Due to the large size of the trained model files, they are hosted externally on Google Drive. Below you will find the links to download each of the model files. Each model corresponds to a specific set of hyperparameters tested during our experiments.

Available Models
Here is a list of the available models along with their corresponding hyperparameters and Google Drive download links:

Model with lr=0.005, batch_size=8, l1_weight=100, dropout_rate=0.1

Description: This model achieved the highest performance in our tests, showing excellent balance between similarity and noise reduction.
Google Drive Link: https://drive.google.com/drive/folders/1INPnHaby9jZZUpLsL0K1skMNQ0YX5jbE?usp=drive_link
Model with lr=0.001, batch_size=8, l1_weight=10, dropout_rate=0.5

Description: Strong in both SSIM and PSNR metrics, excellent for maintaining high image quality.
Google Drive Link: https://drive.google.com/drive/folders/1-0lGv7r1zxtROls2EO8GgpKuT0ckxbaa?usp=drive_link
Model with lr=0.005, batch_size=8, l1_weight=10, dropout_rate=0.5

Description: High SSIM score and decent PSNR, indicating good image similarity with reasonable noise reduction.
Google Drive Link: https://drive.google.com/drive/folders/1-0lGv7r1zxtROls2EO8GgpKuT0ckxbaa?usp=drive_link
How to Use the Models
After downloading the models, please follow these steps to use them in your projects:

Download the Model: Click on the link provided above and download the model file to your local machine.
Place the Model in Your Project Directory: Move the downloaded .pth file into the designated model directory in your project structure.
Update Model Path in Code: Ensure your code references the correct path where the model file is stored.
Load and Use the Model: Use the standard PyTorch method to load the model weights and evaluate or further fine-tune the model on your data.
Troubleshooting
If you encounter any issues while downloading or using the models, please check the following:

Ensure you have sufficient permissions to access the Google Drive links.
Verify that the downloaded files are complete and not corrupted.
Check that your environment meets all the dependencies required to run the model.
