# Trash Classification

A Convolutional Neural Network (CNN) model for classifying trash images into various categories using the TrashNet dataset. This project utilizes TensorFlow and Keras for building the model and Weights & Biases (wandb) for tracking experiments.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

1. **Goal**: Build a CNN model to classify trash images into categories such as glass, paper, metal, plastic, and more.
2. **Frameworks Used**: 
    - TensorFlow/Keras for deep learning model development.
    - Weights & Biases (wandb) for experiment tracking.
3. **Dataset**: The TrashNet dataset is used, which contains images of different types of waste for classification.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/TrashClassification.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd TrashClassification
    ```

3. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

4. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

1. **Download the TrashNet dataset** from Hugging Face:
    - This project uses the `garythung/trashnet` dataset, available via the Hugging Face Datasets library.

2. **Load the dataset using the code**:
    ```python
    from datasets import load_dataset
    dataset = load_dataset("garythung/trashnet")
    ```

3. **Save the dataset locally**:
    - The images will be stored in the `./trashnet_data` directory. The code to load and organize the dataset is included in the project script.

## Model Architecture

1. **Architecture**: This project builds a Convolutional Neural Network (CNN) with the following layers:
    - Convolutional layers with ReLU activation.
    - MaxPooling layers for down-sampling.
    - Dropout layers to prevent overfitting.
    - Dense layers for classification.
    
2. **Model summary**:
    - The CNN model's architecture can be visualized by running `model.summary()` in the provided script.

3. **Modifying the architecture**:
    - The architecture can be adjusted within the script where the model is built (look for the section defining the Sequential model).

## Training

1. **Set up training parameters**:
    - Modify hyperparameters such as batch size, number of epochs, and optimizer directly within the script.

2. **Initialize wandb for tracking**:
    - If you wish to track the experiment using Weights & Biases (wandb), ensure that you have configured wandb correctly with your API key.

3. **Run the training script**:
    - The training code is included in the provided script. You can run it using:
    ```bash
    python your_script_name.py
    ```

4. **Monitor training progress**:
    - Progress will be shown in the terminal, and metrics will be logged to wandb (if configured).

## Evaluation

1. **Evaluate the model on validation data**:
    - The evaluation happens after training. Metrics such as accuracy and loss will be computed for the validation set.

2. **Generate performance plots**:
    - Accuracy and loss plots for both training and validation sets will be generated automatically at the end of the training.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/TrashClassification.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd TrashClassification
    ```

3. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the script**:
    ```bash
    python your_script_name.py
    ```

5. **Track the model performance**:
    - The project logs metrics like accuracy and loss after each epoch to the terminal, and optionally to Weights & Biases.

6. **Modify the model or hyperparameters**:
    - The model architecture and hyperparameters can be adjusted directly in the script.

## Results

1. **View training and validation performance**:
    - Training and validation accuracy and loss graphs will be plotted automatically after training.

2. **Classification results**:
    - The model's performance on trash classification can be evaluated using metrics like accuracy and loss on the validation set.

## Contributing

1. **Fork the repository** to your GitHub account.
2. **Create a new branch** for your changes:
    ```bash
    git checkout -b my-new-feature
    ```
3. **Make changes and commit**:
    ```bash
    git commit -am 'Add some feature'
    ```
4. **Push to the branch**:
    ```bash
    git push origin my-new-feature
    ```
5. **Submit a Pull Request** on GitHub.

