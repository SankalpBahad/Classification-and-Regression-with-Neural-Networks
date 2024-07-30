## Exploring Classification and Regression with Neural Networks

This repository contains the code and analysis for an assignment exploring the application of different neural network architectures for classification and regression problems. 

**Project Structure:**

* **`Classical_Neural_Networks`**: Folder containing the implementation of:
    * **Task 1**: **Multinomial Logistic Regression**:  Implementing a multinomial logistic regression model from scratch using NumPy and Pandas for classifying wine quality.
    * **Task 2**: **MLP Classification**:  Building an MLP classifier from scratch using NumPy and Pandas, experimenting with activation functions and optimization techniques for classifying wine quality.
    * **Task 3**: **MLP Regression**: Implementing an MLP for regression from scratch using NumPy and Pandas to predict housing prices.
* **`Convolutional_and_Advanced_MNIST`**: Folder containing the implementation of:
    * **Task 4**: **CNN Image Classification**: Training a convolutional neural network (CNN) for image classification on the MNIST dataset using PyTorch, including hyperparameter tuning and noise handling.
    * **Task 5**: **Advanced MNIST**:  Exploring variations of the MNIST dataset, including Multi-digit MNIST and Permuted MNIST, using MLP and CNN models.

* **`datasets/`**: Folder containing the datasets used for the assignment:
    * `winequality-red.csv`: Dataset for Task 1, 2, and 3.
    * `advertisement.csv`: Dataset for Task 2 (Multi-label Classification).
    * `housing.csv`: Dataset for Task 3.
    * `mnist-with-awgn.mat`: Noisy MNIST dataset for Task 4.
    * `DoubleMNIST.zip`: Multi-MNIST dataset for Task 5.
    * `PermutedMNIST.zip`: Permuted MNIST dataset for Task 5.

**Instructions:**

1. **Install Required Libraries:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn plotly torch torchvision wandb scipy
   ```
2. **Run the Jupyter Notebooks:**
   ```bash
   jupyter notebook Classical_Neural_Networks/*.ipynb 
   jupyter notebook Convolutional_and_Advanced_MNIST/*.ipynb
   ```
3. **Explore the code:**
   Each notebook provides detailed explanations and code comments, guiding you through the implementation and analysis of each task. 

**Key Concepts:**

* **Multinomial Logistic Regression:**  A generalization of logistic regression for multi-class classification problems.
* **Multi-Layer Perceptron (MLP):** A feedforward neural network with multiple hidden layers, used for classification and regression.
* **Convolutional Neural Networks (CNN):** A type of neural network designed for image processing tasks.
* **Hyperparameter Tuning:** The process of finding the optimal values for hyperparameters that control the behavior of a machine learning model.
* **Weights & Biases (W&B):** A platform for experiment tracking and visualization. 
* **Autoencoders:** Neural networks used for dimensionality reduction and denoising.

**This assignment provides a comprehensive exploration of various neural network architectures and techniques. You can use this repository as a foundation for further learning and experimentation with these powerful tools.**

**Note:** 

This project is for educational purposes and serves as a starting point for further exploration. Feel free to experiment with the code, explore different hyperparameter values, and apply these techniques to different datasets. 




