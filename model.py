import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

def setup_device_and_paths():
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoints/best_model.pth'
    os.makedirs('checkpoints', exist_ok=True)
    return device, checkpoint_path

def initialize_model(device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def load_checkpoint(model, optimizer, checkpoint_path, device):
    start_epoch = 0
    best_val_acc = 0.0
    if os.path.exists(checkpoint_path):
        print("Checkpoint found. Loading model...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_acc = checkpoint['val_acc']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Model loaded. Resuming from epoch {start_epoch}, best validation acc = {best_val_acc:.4f}")
    else:
        print("No checkpoint found. Will train from scratch.")
    return model, optimizer, start_epoch, best_val_acc

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0

def train_model(model, optimizer, criterion, device, checkpoint_path,
                train_loader, val_loader, start_epoch=0, num_epochs=10, best_val_acc=0.0):
    
    early_stopping = EarlyStopping(patience=3, verbose=True)

    if start_epoch < num_epochs:
        print(f"Starting training from epoch {start_epoch} to {num_epochs}.")

        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

            # Validation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            print(f"Validation Accuracy: {val_acc:.4f}")

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'loss': avg_loss
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Best model saved at epoch {epoch+1} with val_acc {val_acc:.4f}")

            # Early Stopping
            early_stopping(val_acc)
            if early_stopping.early_stop:
                print("Early stopping activated!")
                break

        # Save final model
        torch.save(model.state_dict(), 'resnet18_patchcamelyon_final.pth')
        print("Training completed. Final model saved.")
    else:
        print(f"Model already trained up to {start_epoch} epochs. No new training needed.")

    return model

