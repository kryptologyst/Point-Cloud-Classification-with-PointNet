# Project 143: Advanced Point Cloud Classification with PointNet
# Modern implementation with latest tools and techniques

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import os
import json
from datetime import datetime
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the point cloud classification project"""
    num_points: int = 1024
    num_classes: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = "data"
    model_save_path: str = "models"
    results_dir: str = "results"
    
    def __post_init__(self):
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.model_save_path, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)

class MockPointCloudDataset:
    """Mock dataset generator for point cloud classification"""
    
    def __init__(self, num_samples: int = 1000, num_points: int = 1024, num_classes: int = 10):
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes
        self.class_names = [
            'airplane', 'bathtub', 'bed', 'chair', 'dresser', 
            'monitor', 'night_stand', 'sofa', 'table', 'toilet'
        ]
        
    def generate_point_cloud(self, class_id: int) -> np.ndarray:
        """Generate a synthetic point cloud for a given class"""
        if class_id == 0:  # Airplane
            return self._generate_airplane()
        elif class_id == 1:  # Bathtub
            return self._generate_bathtub()
        elif class_id == 2:  # Bed
            return self._generate_bed()
        elif class_id == 3:  # Chair
            return self._generate_chair()
        elif class_id == 4:  # Dresser
            return self._generate_dresser()
        elif class_id == 5:  # Monitor
            return self._generate_monitor()
        elif class_id == 6:  # Night stand
            return self._generate_night_stand()
        elif class_id == 7:  # Sofa
            return self._generate_sofa()
        elif class_id == 8:  # Table
            return self._generate_table()
        else:  # Toilet
            return self._generate_toilet()
    
    def _generate_airplane(self) -> np.ndarray:
        """Generate airplane-like point cloud"""
        points = []
        # Fuselage
        fuselage_length = np.random.uniform(0.8, 1.2)
        fuselage_points = np.random.uniform(-fuselage_length/2, fuselage_length/2, (200, 1))
        fuselage_y = np.random.normal(0, 0.05, (200, 1))
        fuselage_z = np.random.normal(0, 0.05, (200, 1))
        points.extend(np.hstack([fuselage_points, fuselage_y, fuselage_z]))
        
        # Wings
        wing_span = np.random.uniform(0.6, 1.0)
        wing_points = np.random.uniform(-wing_span/2, wing_span/2, (300, 1))
        wing_x = np.random.normal(0, 0.1, (300, 1))
        wing_z = np.random.normal(0, 0.02, (300, 1))
        points.extend(np.hstack([wing_x, wing_points, wing_z]))
        
        # Tail
        tail_points = np.random.uniform(0.3, 0.6, (100, 1))
        tail_y = np.random.normal(0, 0.05, (100, 1))
        tail_z = np.random.uniform(0.1, 0.3, (100, 1))
        points.extend(np.hstack([tail_points, tail_y, tail_z]))
        
        # Add noise and normalize
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _generate_chair(self) -> np.ndarray:
        """Generate chair-like point cloud"""
        points = []
        # Seat
        seat_points = np.random.uniform(-0.3, 0.3, (200, 2))
        seat_z = np.random.uniform(0.3, 0.4, (200, 1))
        points.extend(np.hstack([seat_points, seat_z]))
        
        # Backrest
        backrest_x = np.random.uniform(-0.3, 0.3, (150, 1))
        backrest_y = np.random.uniform(0.2, 0.3, (150, 1))
        backrest_z = np.random.uniform(0.4, 0.8, (150, 1))
        points.extend(np.hstack([backrest_x, backrest_y, backrest_z]))
        
        # Legs
        for _ in range(4):
            leg_x = np.random.uniform(-0.25, 0.25, (50, 1))
            leg_y = np.random.uniform(-0.25, 0.25, (50, 1))
            leg_z = np.random.uniform(0, 0.3, (50, 1))
            points.extend(np.hstack([leg_x, leg_y, leg_z]))
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _generate_table(self) -> np.ndarray:
        """Generate table-like point cloud"""
        points = []
        # Tabletop
        top_points = np.random.uniform(-0.4, 0.4, (300, 2))
        top_z = np.random.uniform(0.7, 0.8, (300, 1))
        points.extend(np.hstack([top_points, top_z]))
        
        # Legs
        for _ in range(4):
            leg_x = np.random.uniform(-0.3, 0.3, (50, 1))
            leg_y = np.random.uniform(-0.3, 0.3, (50, 1))
            leg_z = np.random.uniform(0, 0.7, (50, 1))
            points.extend(np.hstack([leg_x, leg_y, leg_z]))
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _generate_bed(self) -> np.ndarray:
        """Generate bed-like point cloud"""
        points = []
        # Mattress
        mattress_points = np.random.uniform(-0.6, 0.6, (400, 2))
        mattress_z = np.random.uniform(0.2, 0.3, (400, 1))
        points.extend(np.hstack([mattress_points, mattress_z]))
        
        # Headboard
        headboard_x = np.random.uniform(-0.6, 0.6, (100, 1))
        headboard_y = np.random.uniform(0.4, 0.5, (100, 1))
        headboard_z = np.random.uniform(0.3, 0.8, (100, 1))
        points.extend(np.hstack([headboard_x, headboard_y, headboard_z]))
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _generate_sofa(self) -> np.ndarray:
        """Generate sofa-like point cloud"""
        points = []
        # Main body
        body_points = np.random.uniform(-0.8, 0.8, (500, 2))
        body_z = np.random.uniform(0.2, 0.4, (500, 1))
        points.extend(np.hstack([body_points, body_z]))
        
        # Backrest
        backrest_x = np.random.uniform(-0.8, 0.8, (200, 1))
        backrest_y = np.random.uniform(0.6, 0.8, (200, 1))
        backrest_z = np.random.uniform(0.4, 0.8, (200, 1))
        points.extend(np.hstack([backrest_x, backrest_y, backrest_z]))
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _generate_bathtub(self) -> np.ndarray:
        """Generate bathtub-like point cloud"""
        points = []
        # Tub body
        tub_points = np.random.uniform(-0.4, 0.4, (300, 2))
        tub_z = np.random.uniform(0, 0.3, (300, 1))
        points.extend(np.hstack([tub_points, tub_z]))
        
        # Tub walls
        wall_points = np.random.uniform(-0.4, 0.4, (200, 2))
        wall_z = np.random.uniform(0.3, 0.5, (200, 1))
        points.extend(np.hstack([wall_points, wall_z]))
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _generate_dresser(self) -> np.ndarray:
        """Generate dresser-like point cloud"""
        points = []
        # Main body
        body_points = np.random.uniform(-0.4, 0.4, (300, 2))
        body_z = np.random.uniform(0, 0.8, (300, 1))
        points.extend(np.hstack([body_points, body_z]))
        
        # Drawers
        for i in range(3):
            drawer_y = np.random.uniform(-0.3, 0.3, (50, 1))
            drawer_z = np.random.uniform(0.1 + i*0.2, 0.2 + i*0.2, (50, 1))
            drawer_x = np.random.uniform(-0.35, 0.35, (50, 1))
            points.extend(np.hstack([drawer_x, drawer_y, drawer_z]))
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _generate_monitor(self) -> np.ndarray:
        """Generate monitor-like point cloud"""
        points = []
        # Screen
        screen_points = np.random.uniform(-0.3, 0.3, (200, 2))
        screen_z = np.random.uniform(0.4, 0.6, (200, 1))
        points.extend(np.hstack([screen_points, screen_z]))
        
        # Stand
        stand_points = np.random.uniform(-0.1, 0.1, (100, 2))
        stand_z = np.random.uniform(0, 0.4, (100, 1))
        points.extend(np.hstack([stand_points, stand_z]))
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _generate_night_stand(self) -> np.ndarray:
        """Generate night stand-like point cloud"""
        points = []
        # Top surface
        top_points = np.random.uniform(-0.2, 0.2, (100, 2))
        top_z = np.random.uniform(0.4, 0.5, (100, 1))
        points.extend(np.hstack([top_points, top_z]))
        
        # Body
        body_points = np.random.uniform(-0.2, 0.2, (150, 2))
        body_z = np.random.uniform(0, 0.4, (150, 1))
        points.extend(np.hstack([body_points, body_z]))
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _generate_toilet(self) -> np.ndarray:
        """Generate toilet-like point cloud"""
        points = []
        # Bowl
        bowl_points = np.random.uniform(-0.2, 0.2, (200, 2))
        bowl_z = np.random.uniform(0, 0.3, (200, 1))
        points.extend(np.hstack([bowl_points, bowl_z]))
        
        # Tank
        tank_points = np.random.uniform(-0.15, 0.15, (100, 2))
        tank_z = np.random.uniform(0.3, 0.6, (100, 1))
        points.extend(np.hstack([tank_points, tank_z]))
        
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape)
        return self._normalize_points(points)
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere"""
        # Center the points
        points = points - np.mean(points, axis=0)
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the complete dataset"""
        X = []
        y = []
        
        samples_per_class = self.num_samples // self.num_classes
        
        for class_id in range(self.num_classes):
            for _ in range(samples_per_class):
                point_cloud = self.generate_point_cloud(class_id)
                # Sample exactly num_points
                if len(point_cloud) > self.num_points:
                    indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
                    point_cloud = point_cloud[indices]
                elif len(point_cloud) < self.num_points:
                    # Pad with random points if we don't have enough
                    padding = np.random.normal(0, 0.1, (self.num_points - len(point_cloud), 3))
                    point_cloud = np.vstack([point_cloud, padding])
                
                X.append(point_cloud)
                y.append(class_id)
        
        return np.array(X), np.array(y)

class PointCloudAugmentation:
    """Data augmentation techniques for point clouds"""
    
    @staticmethod
    def random_rotation(points: np.ndarray, max_angle: float = 2 * np.pi) -> np.ndarray:
        """Apply random rotation around z-axis"""
        angle = np.random.uniform(0, max_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        return points @ rotation_matrix.T
    
    @staticmethod
    def random_scaling(points: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random scaling"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return points * scale
    
    @staticmethod
    def random_translation(points: np.ndarray, translation_range: float = 0.1) -> np.ndarray:
        """Apply random translation"""
        translation = np.random.uniform(-translation_range, translation_range, 3)
        return points + translation
    
    @staticmethod
    def jitter_points(points: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to points"""
        noise = np.random.normal(0, sigma, points.shape)
        return points + noise
    
    @staticmethod
    def augment_point_cloud(points: np.ndarray) -> np.ndarray:
        """Apply all augmentation techniques"""
        points = PointCloudAugmentation.random_rotation(points)
        points = PointCloudAugmentation.random_scaling(points)
        points = PointCloudAugmentation.random_translation(points)
        points = PointCloudAugmentation.jitter_points(points)
        return points

class TNet(nn.Module):
    """T-Net transformation network for PointNet"""
    
    def __init__(self, k: int = 3):
        super(TNet, self).__init__()
        self.k = k
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x

class PointNet(nn.Module):
    """Complete PointNet architecture with T-Net transformations"""
    
    def __init__(self, num_classes: int = 10, num_points: int = 1024):
        super(PointNet, self).__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        
        # Input transformation
        self.input_transform = TNet(k=3)
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Feature transformation
        self.feature_transform = TNet(k=64)
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input transformation
        trans = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        point_feat = x
        
        # Feature transformation
        trans_feat = self.feature_transform(point_feat)
        x = torch.bmm(point_feat.transpose(2, 1), trans_feat).transpose(2, 1)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Classification head
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x, trans, trans_feat

class PointCloudTrainer:
    """Training and evaluation class for PointNet"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = PointNet(config.num_classes, config.num_points).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output, trans, trans_feat = self.model(data)
            
            # Classification loss
            loss = self.criterion(output, target)
            
            # Regularization loss for transformations
            reg_loss = self._get_regularization_loss(trans, trans_feat)
            total_loss_batch = loss + 0.001 * reg_loss
            
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _, _ = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _get_regularization_loss(self, trans: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        """Calculate regularization loss for transformations"""
        # Ensure transformations are close to orthogonal
        I = torch.eye(trans.size(1), device=trans.device).unsqueeze(0).repeat(trans.size(0), 1, 1)
        loss_trans = F.mse_loss(torch.bmm(trans, trans.transpose(2, 1)), I)
        
        I_feat = torch.eye(trans_feat.size(1), device=trans_feat.device).unsqueeze(0).repeat(trans_feat.size(0), 1, 1)
        loss_feat = F.mse_loss(torch.bmm(trans_feat, trans_feat.transpose(2, 1)), I_feat)
        
        return loss_trans + loss_feat
    
    def train(self, train_loader, val_loader, num_epochs: int = None):
        """Complete training loop"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
            
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save model checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        logger.info("Training completed!")
        self.save_model('final_model.pth')
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        os.makedirs(self.config.model_save_path, exist_ok=True)
        filepath = os.path.join(self.config.model_save_path, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        filepath = os.path.join(self.config.model_save_path, filename)
        
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_losses = checkpoint.get('train_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
            
            logger.info(f"Model loaded from {filepath}")
        else:
            logger.warning(f"Model file {filepath} not found")

class PointCloudVisualizer:
    """Visualization utilities for point clouds"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.colors = px.colors.qualitative.Set3
    
    def plot_point_cloud_3d(self, points: np.ndarray, title: str = "Point Cloud", 
                           color: str = 'blue', size: int = 2) -> go.Figure:
        """Create 3D plot of point cloud using Plotly"""
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                opacity=0.8
            ),
            name=title
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_training_history(self, trainer: PointCloudTrainer) -> go.Figure:
        """Plot training history"""
        epochs = range(1, len(trainer.train_losses) + 1)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Loss', 'Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=trainer.train_losses, name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=trainer.val_losses, name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=trainer.train_accuracies, name='Train Acc', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=trainer.val_accuracies, name='Val Acc', line=dict(color='red')),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Training History",
            width=1000,
            height=400
        )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=self.class_names,
            y=self.class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=600,
            height=500
        )
        
        return fig

def create_streamlit_app():
    """Create Streamlit web application"""
    st.set_page_config(
        page_title="Point Cloud Classification",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Point Cloud Classification with PointNet")
    st.markdown("Advanced 3D point cloud classification using modern PointNet architecture")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model parameters
    num_points = st.sidebar.slider("Number of Points", 512, 2048, 1024)
    num_classes = st.sidebar.slider("Number of Classes", 5, 20, 10)
    batch_size = st.sidebar.slider("Batch Size", 8, 64, 32)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
    num_epochs = st.sidebar.slider("Number of Epochs", 10, 200, 50)
    
    # Create configuration
    config = Config(
        num_points=num_points,
        num_classes=num_classes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Training", "Visualization", "Prediction"])
    
    with tab1:
        st.header("Dataset Generation")
        
        if st.button("Generate Mock Dataset"):
            with st.spinner("Generating dataset..."):
                dataset = MockPointCloudDataset(
                    num_samples=1000,
                    num_points=config.num_points,
                    num_classes=config.num_classes
                )
                X, y = dataset.generate_dataset()
                
                st.success(f"Dataset generated! Shape: {X.shape}")
                
                # Show class distribution
                unique, counts = np.unique(y, return_counts=True)
                fig = px.bar(
                    x=[dataset.class_names[i] for i in unique],
                    y=counts,
                    title="Class Distribution",
                    labels={'x': 'Class', 'y': 'Count'}
                )
                st.plotly_chart(fig)
                
                # Show sample point clouds
                st.subheader("Sample Point Clouds")
                cols = st.columns(3)
                
                for i in range(min(9, config.num_classes)):
                    with cols[i % 3]:
                        class_id = i
                        sample_idx = np.where(y == class_id)[0][0]
                        sample_points = X[sample_idx]
                        
                        visualizer = PointCloudVisualizer(dataset.class_names)
                        fig = visualizer.plot_point_cloud_3d(
                            sample_points,
                            title=f"{dataset.class_names[class_id]}",
                            color=visualizer.colors[i % len(visualizer.colors)]
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Model Training")
        
        if st.button("Start Training"):
            with st.spinner("Training model..."):
                # Generate dataset
                dataset = MockPointCloudDataset(
                    num_samples=1000,
                    num_points=config.num_points,
                    num_classes=config.num_classes
                )
                X, y = dataset.generate_dataset()
                
                # Split dataset
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Convert to tensors
                X_train = torch.FloatTensor(X_train)
                X_val = torch.FloatTensor(X_val)
                y_train = torch.LongTensor(y_train)
                y_val = torch.LongTensor(y_val)
                
                # Create data loaders
                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
                
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
                
                # Initialize trainer
                trainer = PointCloudTrainer(config)
                
                # Train model
                trainer.train(train_loader, val_loader, num_epochs)
                
                st.success("Training completed!")
                
                # Show training history
                visualizer = PointCloudVisualizer(dataset.class_names)
                history_fig = visualizer.plot_training_history(trainer)
                st.plotly_chart(history_fig)
    
    with tab3:
        st.header("Point Cloud Visualization")
        
        # Generate sample point cloud
        if st.button("Generate Random Point Cloud"):
            dataset = MockPointCloudDataset(
                num_samples=1,
                num_points=config.num_points,
                num_classes=config.num_classes
            )
            
            class_id = np.random.randint(0, config.num_classes)
            points = dataset.generate_point_cloud(class_id)
            
            visualizer = PointCloudVisualizer(dataset.class_names)
            fig = visualizer.plot_point_cloud_3d(
                points,
                title=f"Generated {dataset.class_names[class_id]}",
                color='lightblue'
            )
            st.plotly_chart(fig)
    
    with tab4:
        st.header("Model Prediction")
        
        st.info("Upload a point cloud file or use a generated sample for prediction")
        
        # Generate sample for prediction
        if st.button("Generate Sample for Prediction"):
            dataset = MockPointCloudDataset(
                num_samples=1,
                num_points=config.num_points,
                num_classes=config.num_classes
            )
            
            class_id = np.random.randint(0, config.num_classes)
            points = dataset.generate_point_cloud(class_id)
            
            # Visualize the point cloud
            visualizer = PointCloudVisualizer(dataset.class_names)
            fig = visualizer.plot_point_cloud_3d(
                points,
                title="Point Cloud for Prediction",
                color='lightgreen'
            )
            st.plotly_chart(fig)
            
            # Make prediction (if model exists)
            model_path = os.path.join(config.model_save_path, 'final_model.pth')
            if os.path.exists(model_path):
                trainer = PointCloudTrainer(config)
                trainer.load_model('final_model.pth')
                
                # Prepare input
                input_tensor = torch.FloatTensor(points).unsqueeze(0).to(trainer.device)
                
                # Make prediction
                trainer.model.eval()
                with torch.no_grad():
                    output, _, _ = trainer.model(input_tensor)
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = output.argmax(dim=1).item()
                
                st.subheader("Prediction Results")
                st.write(f"**True Class:** {dataset.class_names[class_id]}")
                st.write(f"**Predicted Class:** {dataset.class_names[predicted_class]}")
                
                # Show confidence scores
                conf_df = pd.DataFrame({
                    'Class': dataset.class_names,
                    'Confidence': probabilities[0].cpu().numpy()
                }).sort_values('Confidence', ascending=False)
                
                fig = px.bar(conf_df, x='Class', y='Confidence', title='Prediction Confidence')
                st.plotly_chart(fig)
            else:
                st.warning("No trained model found. Please train a model first.")

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        import pandas as pd
        create_streamlit_app()
    except ImportError:
        print("Streamlit not available. Running basic demo...")
        
        # Basic demo without Streamlit
        config = Config()
        
        print("🧠 Point Cloud Classification Demo")
        print("=" * 50)
        
        # Generate mock dataset
        print("Generating mock dataset...")
        dataset = MockPointCloudDataset(
            num_samples=500,
            num_points=config.num_points,
            num_classes=config.num_classes
        )
        X, y = dataset.generate_dataset()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Classes: {dataset.class_names}")
        
        # Split dataset
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.LongTensor(y_train)
        y_val = torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
        
        # Initialize trainer
        trainer = PointCloudTrainer(config)
        
        # Train model
        print("Starting training...")
        trainer.train(train_loader, val_loader, num_epochs=20)
        
        print("Demo completed!")