# Fruits & Vegetables Recognition System

## Project Overview

This is a machine learning project that uses deep learning to classify fruits and vegetables from images. The system employs a Convolutional Neural Network (CNN) built with TensorFlow/Keras to recognize 36 different types of fruits and vegetables. This project demonstrates the complete machine learning workflow including data preprocessing, model architecture design, training, and evaluation.

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

### Step 4: Download Dataset
The dataset will be automatically downloaded when you run the training notebook:

1. Open `Training_fruit_vegetable.ipynb` in Jupyter Notebook or VS Code
2. The first cell uses kagglehub to download the dataset
3. The dataset path will be stored automatically

### Step 5: Model Training
To train the model:

1. Open `Training_fruit_vegetable.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially from top to bottom
3. The notebook will:
   - Download and load the dataset
   - Preprocess training and validation images
   - Build the CNN architecture
   - Train the model with the training data
   - Save the trained model as `trained_model.h5`
   - Save training history as `training_hist.json`

### Step 6: Model Evaluation and Testing
To evaluate the trained model:

1. Open `Testing_fruit_veg_recognition.ipynb`
2. Run all cells to:
   - Load the trained model
   - Evaluate performance on test data
   - View accuracy metrics and confusion matrix
   - Test predictions on sample images
   - Analyze model performance

## Project Structure

```
Fruits and Vegetable Recognition/
├── README.md                           # Project documentation
├── requirement.txt                     # Python dependencies
├── Training_fruit_vegetable.ipynb      # Model training notebook
├── Testing_fruit_veg_recognition.ipynb # Model evaluation notebook
├── trained_model.h5                    # Trained model file
└── training_hist.json                  # Training history and metrics
```

## Model Architecture

The CNN model is built with the following layers:
- **Convolutional layers** - Extract features from images
- **MaxPooling layers** - Reduce spatial dimensions
- **Dropout layers** - Prevent overfitting
- **Flatten layer** - Convert 2D features to 1D
- **Dense layers** - Fully connected layers for classification
- **Output layer** - 36 units with softmax activation

**Model Specifications:**
- **Input Shape**: 64x64x3 (RGB images)
- **Output**: 36 classes (multi-class classification)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

## Training Results

The model is trained on the Kaggle Fruit and Vegetable Image Recognition dataset with:
- Image preprocessing and normalization
- Batch size: 32
- Image size: 64x64 pixels
- Training and validation split

Results and performance metrics can be viewed in the `Testing_fruit_veg_recognition.ipynb` notebook.

## Key Features

- **Automated Dataset Download** - Uses kagglehub for seamless dataset acquisition
- **Data Preprocessing** - Handles image loading, resizing, and normalization
- **CNN Architecture** - Custom-built convolutional neural network
- **Training Visualization** - Plots training/validation accuracy and loss
- **Model Evaluation** - Comprehensive testing with metrics and predictions
- **Model Persistence** - Saves trained model for future use

## Troubleshooting

### Common Issues

1. **ValueError: Shapes incompatible**
   - Ensure the output layer has the correct number of units (36 for this dataset)
   - Verify that `label_mode="categorical"` is set in data loading

2. **Module not found errors**
   - Make sure all dependencies are installed: `pip install -r requirement.txt`
   - Verify you're using the correct Python environment

3. **Dataset download issues**
   - Check your internet connection
   - Ensure kagglehub is properly installed
   - Verify Kaggle API credentials if required

4. **Out of memory errors**
   - Reduce batch size in the training notebook
   - Close other applications to free up RAM
   - Consider using a machine with more memory

### Performance Tips

- **Use GPU acceleration** - Install GPU-enabled TensorFlow for faster training
- **Adjust batch size** - Increase if you have sufficient memory
- **Monitor training** - Watch for overfitting in training/validation curves
- **Save checkpoints** - Save model at regular intervals during long training sessions

## Future Improvements

Potential enhancements for this project:
- Implement data augmentation (rotation, flipping, zooming) for better generalization
- Experiment with transfer learning using pre-trained models (VGG16, ResNet, etc.)
- Add more fruit/vegetable classes to expand recognition capabilities
- Implement early stopping and learning rate scheduling
- Cross-validation for more robust model evaluation
- Deploy the model as a web service or mobile application

## Dataset Credit

This project uses the **Fruit and Vegetable Image Recognition** dataset from Kaggle:
- **Source**: [Kaggle - Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
- **Creator**: Kritik Seth

## License

This project is just to survive the AI class of my ITE-Y4-S1, educational purposes. Please respect the dataset license terms when using the Kaggle dataset.
