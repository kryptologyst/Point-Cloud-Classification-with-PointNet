# Point Cloud Classification with PointNet

Advanced 3D point cloud classification using modern PointNet architecture with comprehensive visualization and web interface.

## Features

- **Modern PointNet Implementation**: Complete PointNet architecture with T-Net transformations
- **Mock Dataset Generator**: Synthetic point cloud generation for 10 object classes
- **Data Augmentation**: Advanced augmentation techniques for point clouds
- **Interactive Web UI**: Streamlit-based interface for training and visualization
- **3D Visualization**: Interactive 3D point cloud visualization with Plotly
- **Model Management**: Save/load trained models with checkpoints
- **Training Metrics**: Comprehensive training history and validation metrics

## Object Classes

The model can classify the following 10 object categories:
- ✈️ Airplane
- 🛁 Bathtub  
- 🛏️ Bed
- 🪑 Chair
- 🗄️ Dresser
- 🖥️ Monitor
- 🛏️ Night Stand
- 🛋️ Sofa
- 🪑 Table
- 🚽 Toilet

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd 0143_Point_cloud_classification
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run 0143.py
```

## Usage

### Web Interface (Recommended)

Launch the Streamlit web application:
```bash
streamlit run 0143.py
```

The web interface provides four main tabs:

1. **Dataset**: Generate and visualize mock point cloud datasets
2. **Training**: Train the PointNet model with configurable parameters
3. **Visualization**: Interactive 3D point cloud visualization
4. **Prediction**: Make predictions on new point clouds

### Command Line Interface

Run the basic demo without Streamlit:
```bash
python 0143.py
```

## Architecture

### PointNet Model

The implementation includes:

- **Input Transformation Network (T-Net)**: Learns optimal input transformations
- **Feature Transformation Network**: Transforms learned features
- **Shared MLP Layers**: Extract point-wise features
- **Global Max Pooling**: Permutation-invariant aggregation
- **Classification Head**: Final classification layers

### Key Components

- `MockPointCloudDataset`: Generates synthetic point clouds for training
- `PointCloudAugmentation`: Data augmentation techniques
- `PointCloudTrainer`: Complete training pipeline
- `PointCloudVisualizer`: 3D visualization utilities
- `Config`: Centralized configuration management

## Training Configuration

Default training parameters:
- **Number of Points**: 1024 per point cloud
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Number of Epochs**: 100
- **Optimizer**: Adam
- **Scheduler**: StepLR (step_size=30, gamma=0.5)

## Visualization Features

- **3D Point Cloud Rendering**: Interactive 3D visualization with Plotly
- **Training History**: Loss and accuracy curves
- **Confusion Matrix**: Classification performance visualization
- **Class Distribution**: Dataset statistics
- **Prediction Confidence**: Probability scores for each class

## 📁 Project Structure

```
0143_Point_cloud_classification/
├── 0143.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Dataset storage (auto-created)
├── models/                # Saved models (auto-created)
└── results/               # Training results (auto-created)
```

## 🔧 Advanced Features

### Data Augmentation
- Random rotation around Z-axis
- Random scaling
- Random translation
- Gaussian noise jittering

### Model Regularization
- Transformation matrix regularization
- Dropout for overfitting prevention
- Batch normalization for stable training

### Training Features
- Automatic checkpointing
- Learning rate scheduling
- Comprehensive logging
- Validation metrics tracking

## Quick Start

1. **Generate Dataset**:
   - Open the web interface
   - Go to "Dataset" tab
   - Click "Generate Mock Dataset"

2. **Train Model**:
   - Go to "Training" tab
   - Adjust parameters if needed
   - Click "Start Training"

3. **Visualize Results**:
   - Go to "Visualization" tab
   - Generate random point clouds
   - Explore 3D visualizations

4. **Make Predictions**:
   - Go to "Prediction" tab
   - Generate sample for prediction
   - View classification results

## Performance

The model typically achieves:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Training Time**: ~5-10 minutes (20 epochs, 1000 samples)

## Technical Details

### Point Cloud Processing
- **Normalization**: Points centered and scaled to unit sphere
- **Sampling**: Fixed number of points per cloud (1024)
- **Augmentation**: Multiple transformation techniques

### Model Architecture
- **Input**: (B, N, 3) point coordinates
- **T-Net**: 3x3 transformation matrix learning
- **MLP**: 64 → 128 → 1024 feature dimensions
- **Output**: (B, num_classes) classification logits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the "1000 AI Projects" series and is available under the MIT License.

## Acknowledgments

- Original PointNet paper by Qi et al.
- PyTorch Geometric for point cloud utilities
- Streamlit for web interface
- Plotly for 3D visualization

## Support

For questions or issues:
1. Check the documentation
2. Review the code comments
3. Open an issue on GitHub

 
# Point-Cloud-Classification-with-PointNet
