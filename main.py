import torch
import yaml

from src.data.dataset import get_dataloaders
from src.models.cnn import SimpleCNN
from src.training.engine import train
from src.evaluation.evaluate import evaluate

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_dataloaders(config["batch_size"])

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(config["epochs"]):
    loss = train(model, train_loader, optimizer, criterion, device)
    acc = evaluate(model, test_loader, device)

    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")