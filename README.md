# Trash Classification with CNN

This project demonstrates the implementation of a Convolutional Neural Network (CNN) model to classify images into different trash categories using the **TrashNet** dataset. The model is built using **TensorFlow/Keras** and achieves ~80% accuracy after 25 epochs.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

The goal of this project is to build a model that can classify waste into categories like cardboard, glass, metal, paper, plastic, and trash. Waste classification is important in recycling and waste management.

## Dataset

We are using the **[TrashNet](https://github.com/garythung/trashnet)** dataset, which contains 2527 images of trash, divided into 6 classes:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

To load the dataset, we used the `datasets` library:

```python
from datasets import load_dataset
dataset = load_dataset("garythung/trashnet")
