
# Signature Verification Project

## Link to our Saved Model 
https://drive.google.com/file/d/1vhKQx_Ica7oNLSOdSrcOKVOBQXdBpZ3r/view?usp=sharing

## Introduction

This project focuses on developing a robust signature verification system using Siamese Neural Networks to accurately classify genuine and forged signatures. The system is designed to enhance security in applications such as banking, legal documentation, and identity verification.

## Problem Statement

The primary challenge in signature verification lies in the high variability of handwriting styles and the subtle differences between genuine and forged signatures. Traditional methods often fall short in accuracy and efficiency. Our solution aims to overcome these challenges by leveraging neural networks, particularly Siamese networks, to learn unique features and patterns in signatures.

## Goal

The goal of this project is to create a reliable system for authenticating signatures, capable of accurately distinguishing between genuine and forged signatures. A user-friendly interface has also been implemented to allow real-time verification of uploaded signatures.

## Proposed Solution

We chose Siamese neural networks due to their ability to learn and differentiate subtle differences in pairs of inputs, making them ideal for signature verification. This architecture enhances accuracy and robustness, ensuring reliable performance in real-world applications.

## Technology Stack

- **Frontend:** React, Tailwind CSS, HeadlessUI
- **Backend:** Flask (Python)
- **Database:** MongoDB
- **Machine Learning Model:** Siamese Neural Network

## Project Flow

1. **Dataset:**
   - Source: [Kaggle Signature Matching Dataset](https://www.kaggle.com/datasets/mallapraveen/signature-matching)
   - Training Dataset: 18,000 sets of signature images (genuine and forged)
   - Testing Dataset: 4,500 sets of signature images

2. **Data Preprocessing:**
   - Background removal, resizing, centering, grayscale conversion, and tensor conversion.

3. **Model Training and Testing:**
   - **Architecture:** SigNet as the base network within the Siamese architecture.
   - **Training:** Contrastive loss and binary cross-entropy loss were used. The model was trained over 20 epochs.
   - **Testing:** Evaluation of model performance using cosine similarity, confusion matrix, precision-recall curve, and ROC curves.

4. **Integration:**
   - **Backend:** Flask server for handling API requests and model inference.
   - **Frontend:** React-based interface for user registration, signature upload, and displaying verification results.
   - **Database:** MongoDB for efficient data storage and management.

## Evaluation Results

- **Model Performance:** The Siamese Neural Network showed significant improvements in accuracy, with high precision and recall rates.
- **Comparison with Other Models:**
  - **SVM:** Accuracy of 78%, precision of 75%, recall of 76%.
  - **HMM:** Accuracy of 75%, precision of 72%, recall of 73%.
  - **Neural Networks:** Accuracy of 81%, precision of 78%, recall of 79%.

## Task Distribution

### Sahaj Totla
- **Model Development:** Led the development of the Siamese Neural Network and SigNet network.
- **Model Architecture:** Designed the neural network architecture.
- **Backend Support:** Contributed to the setup and management of the Flask server.

### Hrithik Bagane
- **Frontend Development:** Designed and implemented the user interface using React.
- **Backend Development:** Set up and developed the Flask backend server, database, and API endpoints.
- **Model Testing, Evaluation, and Inference:** Conducted extensive testing and evaluation of the Siamese Neural Network model.

### Kapil Gulani
- **Data Preprocessing:** Handled preprocessing of the dataset.
- **Frontend Support:** Assisted in the development of the frontend interface.
- **Database Management:** Managed the MongoDB database, designed the schema, and set up collections.
