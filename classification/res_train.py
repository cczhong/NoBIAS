import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import time

# --- Configuration ---
IMAGE_DATA_DIR = "output_images_projected3D_128x3_final_bonds"
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 384
BATCH_SIZE = 32
EPOCHS = 30 # Increased epochs since we have a scheduler
LEARNING_RATE = 0.001
MODEL_FILENAME = "resnet18_classifier_stable.pth"

# --- Dataset Definition ---
class InteractionImageDataset(Dataset):
    def __init__(self, all_files, all_labels, transform=None):
        self.file_paths = all_files
        self.labels = all_labels
        self.transform = transform
        self.class_map = {label: i for i, label in enumerate(np.unique(all_labels))}
        self.class_names = list(self.class_map.keys())

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.file_paths[idx]).convert("RGB")
            label_name = self.labels[idx]
            label_idx = self.class_map[label_name]
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label_idx, dtype=torch.long)
        except (IOError, OSError) as e:
            print(f"Warning: Skipping corrupted image {self.file_paths[idx]}: {e}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

# --- Evaluation Function for Epoch Summaries ---
def evaluate_epoch(model, data_loader, device, class_names, criterion):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            if inputs is None: continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            total_loss += loss.item()
            preds = torch.sigmoid(outputs).squeeze() > 0.5
            all_preds.extend(preds.cpu().numpy().astype(int))
            all_true.extend(labels.cpu().numpy().astype(int))
            
    if not all_true: return 0.0, ""
    
    avg_loss = total_loss / len(data_loader)
    report_dict = classification_report(all_true, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    pair_acc = report_dict[class_names[0]]['recall'] * 100
    stack_acc = report_dict[class_names[1]]['recall'] * 100
    pair_f1 = report_dict[class_names[0]]['f1-score']
    stack_f1 = report_dict[class_names[1]]['f1-score']
    
    summary = f"Val Loss: {avg_loss:.4f} | Acc: P={pair_acc:.1f}% (f1={pair_f1:.2f}), S={stack_acc:.1f}% (f1={stack_f1:.2f})"
    return avg_loss, summary

# --- Main Training Function ---
def train_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isdir(IMAGE_DATA_DIR):
        print(f"ERROR: Image data directory not found: '{IMAGE_DATA_DIR}'")
        return

    all_files, all_labels = [], []
    class_names = sorted([d for d in os.listdir(IMAGE_DATA_DIR) if os.path.isdir(os.path.join(IMAGE_DATA_DIR, d))])
    for class_name in class_names:
        class_dir = os.path.join(IMAGE_DATA_DIR, class_name)
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith('.png')]
        all_files.extend(files)
        all_labels.extend([class_name] * len(files))
    
    indices = np.arange(len(all_files))
    train_indices, temp_indices, _, y_temp = train_test_split(indices, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    val_indices, test_indices, _, _ = train_test_split(temp_indices, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    full_dataset = InteractionImageDataset(all_files, all_labels)
    class_names = full_dataset.class_names
    
    train_dataset = Subset(full_dataset, train_indices)
    train_dataset.dataset.transform = train_transforms
    val_dataset = Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = val_test_transforms
    test_dataset = Subset(full_dataset, test_indices)
    test_dataset.dataset.transform = val_test_transforms

    num_workers = 2 # Safe default for Colab
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print("Initializing ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    print(f"Starting model training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in pbar:
            if inputs is None: continue
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        val_loss, val_summary = evaluate_epoch(model, val_loader, device, class_names, criterion)
        epoch_duration = time.time() - start_time
        
        print(f"EPOCH {epoch+1}/{EPOCHS} SUMMARY | Duration: {epoch_duration:.1f}s | {val_summary}")
        scheduler.step(val_loss)

    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f"Training finished. Final model saved to {MODEL_FILENAME}")
    
    # --- Final Evaluation on Test Set ---
    print("\n--- FINAL TEST SET EVALUATION ---")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            if inputs is None: continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).squeeze() > 0.5
            all_preds.extend(preds.cpu().numpy().astype(int))
            all_true.extend(labels.cpu().numpy().astype(int))
    
    # <<< RE-INTRODUCED: Final, detailed classification report >>>
    print("\nClassification Report (sklearn):")
    report = classification_report(all_true, all_preds, target_names=class_names, zero_division=0)
    print(report)


if __name__ == "__main__":
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
    except ImportError:
        print("Please install scikit-learn for data splitting and reporting: pip install scikit-learn")
    else:
        train_resnet()