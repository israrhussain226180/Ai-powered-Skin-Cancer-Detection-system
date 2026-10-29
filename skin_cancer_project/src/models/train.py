import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
from tqdm import tqdm

import sys
from pathlib import Path

# Add the project root to sys.path to handle imports
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

# Now import using the full path
from src.models.model import get_model
from src.data.loader import get_data_loaders

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # History for plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch}")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

if __name__ == "__main__":
    # --- Hyperparameters & Config ---
    DATA_DIR = os.path.join(project_root, "data/splits")
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LR = 0.001
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    dataloaders, dataset_sizes, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # 2. Initialize Model
    model = get_model(num_classes=num_classes)
    model = model.to(DEVICE)

    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Start Training
    model, history = train_model(
        model, dataloaders, dataset_sizes, criterion, 
        optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS, device=DEVICE
    )

    # 5. Save the final best model
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), 'models/best_efficientnet.pth')
    print("Model saved to models/best_efficientnet.pth")
