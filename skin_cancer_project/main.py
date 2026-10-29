import os
import argparse
from src.models.train import train_model, get_model
from src.data.loader import get_data_loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

def main():
    parser = argparse.ArgumentParser(description="Skin Cancer Classification Project")
    parser.add_argument("--data_dir", type=str, default="data/splits", help="Path to split dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", type=str, default="efficientnet_b0", help="Model name")
    
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    dataloaders, dataset_sizes, class_names = get_data_loaders(args.data_dir, args.batch_size)
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # 2. Initialize Model
    model = get_model(num_classes=num_classes, model_name=args.model)
    model = model.to(device)

    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Train
    model, history = train_model(
        model, dataloaders, dataset_sizes, criterion, 
        optimizer, exp_lr_scheduler, num_epochs=args.epochs, device=device
    )

    # 5. Save
    if not os.path.exists("models"):
        os.makedirs("models")
    save_path = os.path.join("models", f"best_{args.model}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()
