# Autoencoder-Based Image Compression for Handwritten Arabic Digits

## Project Overview
This project builds and evaluates a deep learning **autoencoder** for image compression and reconstruction using a dataset of **handwritten Arabic digit images**. The goal is to learn a compact latent representation of each image while preserving enough information to reconstruct the original digit with good visual quality.

The notebook covers the full workflow from data loading and normalization to model building, training, reconstruction visualization, and memory compression analysis.

## Dataset
The project uses the file **`csvImages 10k x 784.csv`**, which contains:

- **10,000 grayscale images**
- Each image stored as a flattened vector of **784 pixel values**
- Original image size: **28 × 28**
- Pixel range before normalization: **0 to 255**

After loading, the data is normalized to the range **[0, 1]** for neural network training.

## Project Workflow

### 1. Data Loading
The dataset is loaded into a Pandas DataFrame and converted to a NumPy array.

### 2. Data Normalization
Pixel values are scaled by dividing by 255 so the model can train more effectively.

### 3. Image Visualization
The first few samples are reshaped from 784-dimensional vectors into **28 × 28** images and displayed to confirm that the data was loaded correctly.

### 4. Reshaping
Although images are visualized in 2D form, they are reshaped back into **1D vectors of length 784** before being passed into the autoencoder.

### 5. Autoencoder Construction
A fully connected autoencoder is built using Keras.

#### Initial model architecture
- **Input layer:** 784 units
- **Encoder:** Dense(256, ReLU) → Dense(64, ReLU)
- **Decoder:** Dense(256, ReLU) → Dense(784, Sigmoid)

### 6. Model Compilation and Training
The autoencoder is compiled with:

- **Optimizer:** Adam
- **Loss function:** Binary crossentropy

Initial training setup:
- **Epochs:** 20
- **Batch size:** 256
- **Validation split:** 0.2
- **Shuffle:** True

### 7. Encoder Extraction
After training, the encoder portion of the model is separated so that the compressed latent representation can be generated independently.

### 8. Reconstruction Visualization
The notebook compares:

- Original images
- Encoded representations
- Reconstructed images

This helps evaluate how well the model preserves digit structure after compression.

### 9. Hyperparameter Tuning
The model is retrained with different latent sizes to balance compression and reconstruction quality.

Tested encoding sizes include:
- **128**
- **64**
- **32**

The notebook concludes that an encoding size of **64** provides a strong balance between compactness and visual reconstruction quality.

## Results

### Reconstruction Quality
The reconstructed digit images remain visually similar to the originals, preserving the main stroke shapes and overall digit identity. Larger latent dimensions improve image quality, while smaller latent dimensions increase compression but reduce detail.

### Training Behavior
Both training and validation loss decrease steadily, showing that the autoencoder learns meaningful image features without severe overfitting.

### Compression Performance
Each original image contains **784 floating-point values**.

- Original storage: **784 × 4 = 3136 bytes**
- Encoded storage with 64 features: **64 × 4 = 256 bytes**
- Memory saved: **2880 bytes per image**
- Compression savings: **about 91.8%**

This means the learned representation dramatically reduces storage requirements while still allowing usable image reconstruction.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- TensorFlow / Keras

## How to Run
1. Clone this repository.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib tensorflow
   ```
3. Place the dataset file `csvImages 10k x 784.csv` in the project folder.
4. Open the notebook:
   ```bash
   jupyter notebook Autoencoders.ipynb
   ```
5. Run all cells in order.

## Learning Outcomes
This project demonstrates how autoencoders can be used for:

- Unsupervised feature learning
- Dimensionality reduction
- Image compression
- Reconstruction quality analysis
- Trade-off analysis between compression size and information retention

## Possible Improvements
Future enhancements could include:

- Using **convolutional autoencoders** instead of fully connected layers
- Adding regularization such as dropout or L1/L2 penalties
- Comparing multiple optimizers and loss functions
- Evaluating reconstruction quality with quantitative metrics such as MSE or SSIM
- Testing the encoder output for downstream classification tasks

## Repository Structure
```text
.
├── Autoencoders.ipynb
├── csvImages 10k x 784.csv
└── README_Autoencoders.md
```

## Conclusion
This project shows that an autoencoder can successfully compress handwritten Arabic digit images into a much smaller latent space while still reconstructing recognizable outputs. With a **64-dimensional encoding**, the model achieves a practical compromise between image quality and compression efficiency, reducing storage by **approximately 91.8%**.
