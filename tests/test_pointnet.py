import pytest
import torch
import numpy as np
from 0143 import MockPointCloudDataset, PointNet, PointCloudTrainer, Config

def test_mock_dataset_generation():
    """Test mock dataset generation"""
    dataset = MockPointCloudDataset(num_samples=100, num_points=512, num_classes=5)
    X, y = dataset.generate_dataset()
    
    assert X.shape == (100, 512, 3), f"Expected shape (100, 512, 3), got {X.shape}"
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"
    assert len(np.unique(y)) == 5, f"Expected 5 unique classes, got {len(np.unique(y))}"

def test_pointnet_model():
    """Test PointNet model initialization and forward pass"""
    model = PointNet(num_classes=10, num_points=1024)
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1024, 3)
    
    output, trans, trans_feat = model(input_tensor)
    
    assert output.shape == (batch_size, 10), f"Expected output shape (4, 10), got {output.shape}"
    assert trans.shape == (batch_size, 3, 3), f"Expected trans shape (4, 3, 3), got {trans.shape}"
    assert trans_feat.shape == (batch_size, 64, 64), f"Expected trans_feat shape (4, 64, 64), got {trans_feat.shape}"

def test_trainer_initialization():
    """Test trainer initialization"""
    config = Config(num_classes=5, num_points=512, batch_size=16)
    trainer = PointCloudTrainer(config)
    
    assert trainer.model.num_classes == 5
    assert trainer.model.num_points == 512
    assert trainer.device.type in ['cpu', 'cuda']

def test_point_cloud_augmentation():
    """Test point cloud augmentation"""
    from 0143 import PointCloudAugmentation
    
    points = np.random.randn(100, 3)
    
    # Test individual augmentation methods
    rotated = PointCloudAugmentation.random_rotation(points)
    scaled = PointCloudAugmentation.random_scaling(points)
    translated = PointCloudAugmentation.random_translation(points)
    jittered = PointCloudAugmentation.jitter_points(points)
    
    assert rotated.shape == points.shape
    assert scaled.shape == points.shape
    assert translated.shape == points.shape
    assert jittered.shape == points.shape
    
    # Test combined augmentation
    augmented = PointCloudAugmentation.augment_point_cloud(points)
    assert augmented.shape == points.shape

if __name__ == "__main__":
    pytest.main([__file__])
