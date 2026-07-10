# CNN Models for Image Classification 🌺

Welcome to the **cnn-models** repository! This project focuses on building, training, and deploying image classification models using Convolutional Neural Networks (CNNs) and Transfer Learning techniques. 

The models in this repository were trained using a **Kaggle dataset** (specifically focused on flower recognition, recognizing classes like tulips and sunflowers).

---

## 📊 Models and Performance

This project experiments with different deep learning approaches to achieve the best classification accuracy. The following pre-trained models are included in this repository:

*   **Feature Extracted CNN:** 94% Accuracy (`feature_extracted_cnn_accuracy_94.keras`)
*   **Transfer Learning Model v2:** 93% Accuracy (`transfer_learning_accuracy_93.keras`)
*   **Transfer Learning Model v1:** 92% Accuracy (`transfer_learning_accuracy_92.keras`)
*   **Custom CNN Base Model:** 84% Accuracy (`cnn_accuracy_84.keras`)

---

## 🗂️ Project Structure

*   `app.py`: The main application script used to serve/run the model predictions.
*   `extract-datas.ipynb`: Jupyter Notebook containing the data extraction, preprocessing, and model training pipeline.
*   `classes.json`: A JSON file mapping the numerical class indices to their respective human-readable labels.
*   `*.keras`: Saved Keras models with various accuracy levels.
*   `requirements.txt`: A list of all Python dependencies required to run the project.
*   `sunflower.jpg` & `tulip.jpg`: Sample images used for testing model predictions.
*   `.devcontainer/`: Configuration for using this project within a Docker-based development container.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Make sure you have Python installed. It is recommended to use a virtual environment.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Y-Winee/cnn-models.git](https://github.com/Y-Winee/cnn-models.git)
   cd cnn-models

2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt
3. Usage
   ```bash
    python app.py


