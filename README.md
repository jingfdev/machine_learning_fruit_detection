# Fruits & Vegetables Recognition System

## Project Overview

This project is a machine learning application that uses deep learning to classify fruits and vegetables from images. The system employs a Convolutional Neural Network (CNN) built with TensorFlow/Keras to recognize 36 different types of fruits and vegetables. The project includes both training notebooks for model development and a Streamlit web application for real-time prediction. 

### Supported Classes
The model can recognize the following 36 fruits and vegetables:
- **Fruits**: apple, banana, grapes, kiwi, lemon, mango, orange, paprika, pear, pineapple, pomegranate, watermelon
- **Vegetables**: beetroot, bell pepper, cabbage, capsicum, carrot, cauliflower, chilli pepper, corn, cucumber, eggplant, garlic, ginger, jalepeno, lettuce, onion, peas, potato, raddish, soy beans, spinach, sweetcorn, sweetpotato, tomato, turnip

## Requirements

### System Requirements
- Python 3.8 or higher
- Windows/Linux/macOS
- Minimum 4GB RAM (8GB recommended)
- GPU support (optional, for faster training)

### Python Dependencies
```
tensorflow==2.10.0
scikit-learn==1.3.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.13.0
pandas==2.1.0
streamlit
librosa==0.10.1
kagglehub
```

## Project Setup Guide

### Step 1: Clone/Download the Project
Download or clone this repository to your local machine.

### Step 2: Create Virtual Environment (Recommended)
```cmd
# Create a new conda environment
conda create -n fruit-veg-env python=3.9

# Activate the environment
conda activate fruit-veg-env

# Or using venv
python -m venv fruit-veg-env
fruit-veg-env\Scripts\activate
```

### Step 3: Install Dependencies
```cmd
# Navigate to the project directory
cd "Fruits and Vegetable Recognition"

# Install required packages
pip install -r requirement.txt
```

### Step 4: Download Dataset (For Training)
If you want to train the model from scratch:

1. Run the training notebook `Training_fruit_vegetable.ipynb`
2. The dataset will be automatically downloaded using kagglehub
3. The notebook will handle data preprocessing and model training

### Step 5: Model Training (Optional)
If you want to train your own model:

1. Open `Training_fruit_vegetable.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially
3. The trained model will be saved as `trained_model.h5`
4. Training history will be saved as `training_hist.json`

### Step 6: Testing the Model (Optional)
To test the trained model:

1. Open `Testing_fruit_veg_recognition.ipynb`
2. Run the cells to evaluate model performance
3. View accuracy metrics and prediction results

### Step 7: Run the Web Application
```cmd
# Navigate to the webapp directory
cd Fruit_veg_webapp

# Run the Streamlit application
streamlit run main.py
```

The web application will open in your default browser at `http://localhost:8501`

## How to Use the Web Application

### Home Page
- Displays the main interface with project title and overview image

### About Project
- Contains information about the dataset and project details
- Lists all supported fruit and vegetable classes

### Prediction Page
- Upload an image of a fruit or vegetable
- Click "Predict" to get the classification result
- View the predicted class with confidence

## Project Structure

```
Fruits and Vegetable Recognition/
├── README.md                           # Project documentation
├── requirement.txt                     # Python dependencies
├── Training_fruit_vegetable.ipynb      # Model training notebook
├── Testing_fruit_veg_recognition.ipynb # Model testing notebook
├── trained_model.h5                    # Pre-trained model file
├── training_hist.json                  # Training history
├── Fruit_veg_webapp/                   # Web application
│   ├── main.py                         # Streamlit app main file
│   ├── labels.txt                      # Class labels
│   ├── home_img.jpg                    # Home page image
│   └── Download_image/                 # Sample test images
│       ├── Image_1.jpg
│       ├── Image_2.jpg
│       └── ...
```

## Model Architecture

The CNN model includes:
- Convolutional layers for feature extraction
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax activation for multi-class prediction

**Input Shape**: 64x64x3 (RGB images)
**Output**: 36 classes (fruits and vegetables)

## License

This project is just to survive the AI class of my ITE-Y4-S1, educational purposes. Please respect the dataset license terms when using the Kaggle dataset.
