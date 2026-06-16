import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DATA_CSV = r'c:\Users\email\Desktop\Python_AL\Denim_Jeans_ML_PR\ProcessedData\labels.csv'
MODEL_SAVE_PATH = r'c:\Users\email\Desktop\Python_AL\Denim_Jeans_ML_PR\ProcessedData\defect_model.pth'
GRADCAM_SAVE_DIR = r'c:\Users\email\Desktop\Python_AL\Denim_Jeans_ML_PR\ProcessedData\GradCAM_Samples'
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(GRADCAM_SAVE_DIR):
    os.makedirs(GRADCAM_SAVE_DIR)

class DenimDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        label = self.dataframe.loc[idx, 'is_damaged']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Fallback to black image if corrupted
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), img_path

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        # Forward pass
        output = self.model(x)
        self.model.zero_grad()
        # Backward pass on the objective (assuming binary class 1 is the target)
        # We want to see what makes the model predict "damaged"
        output.backward(retain_graph=True)
        
        # Determine weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Multiply activations by weights
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam) # ReLU on CAM
        
        # Normalize
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        
        return cam.cpu().detach().numpy()

def generate_cam_overlay(img_tensor, cam, save_path):
    # Unnormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img_tensor.permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)
    
    # Resize CAM to match image
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = np.uint8(255 * cam)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    
    # Overlay
    cam_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Save comparing original and heatmap side-by-side
    save_img = np.hstack((img[:, :, ::-1], cam_img)) # Convert RGB to BGR for cv2
    cv2.imwrite(save_path, save_img)

def main():
    print(f"Using device: {DEVICE}")
    
    # Load Data
    df = pd.read_csv(DATA_CSV)
    
    # Because of massive imbalance (118 vs 10), we will augment and use weighted loss or stratify
    # To keep it simple, we stratify
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['is_damaged'], random_state=42)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Transforms (using standard ImageNet normalization)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DenimDataset(train_df, train_transforms)
    val_dataset = DenimDataset(val_df, val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    model = model.to(DEVICE)
    
    # Calculate pos_weight for BCEWithLogitsLoss because of class imbalance (Damaged=1)
    num_neg = len(train_df[train_df['is_damaged'] == 0])
    num_pos = len(train_df[train_df['is_damaged'] == 1])
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(DEVICE) if num_pos > 0 else None
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_loss = float('inf')
    
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds == labels).sum().item()
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save Best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" -> Best model saved at Epoch {epoch+1}")

    print("\n--- Training Complete ---")
    
    # Setup Grad-CAM with the best model
    print("Generating sample Grad-CAM Localizations...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # Grad-CAM on MobileNetV2 uses the last convolutional layer features
    target_layer = model.features[-1] 
    grad_cam = GradCAM(model, target_layer)
    
    samples_generated = 0
    # Run on validation set to generate a few Grad-CAM examples
    for images, labels, paths in val_loader:
        images = images.to(DEVICE)
        
        for i in range(images.size(0)):
            img_tensor = images[i].unsqueeze(0)
            label = labels[i].item()
            path = paths[i]
            basename = os.path.basename(path)
            
            # Require requires_grad to be True on the input tensor for Grad-CAM
            img_tensor.requires_grad = True
            
            # Predict
            cam = grad_cam(img_tensor)
            
            save_path = os.path.join(GRADCAM_SAVE_DIR, f"cam_{int(label)}_{basename}")
            generate_cam_overlay(images[i].cpu(), cam, save_path)
            
            samples_generated += 1
            if samples_generated >= 5: # Just save 5 for inspection
                print(f"Saved {samples_generated} Grad-CAM samples to {GRADCAM_SAVE_DIR}")
                return

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') # Ignore PyTorch module hook warnings
    main()
