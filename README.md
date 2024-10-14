SketchGAN is a deep learning-based project that focuses on generating realistic images from sketches using a Generative Adversarial Network (GAN) model. This project is designed to aid in criminal identification by generating possible criminal likenesses based on witness or police sketches. It leverages GANs, specifically Conditional GANs (cGANs), to generate high-quality images conditioned on input sketches.

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

