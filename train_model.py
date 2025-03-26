import pandas as pd
import torch
import torch.nn as nn
from model import TitanicModel
from torch.utils.data import TensorDataset, DataLoader

def test_train_split(data : torch.utils.data.Dataset, seed : int) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Split the training data into train and validation'''
    generator = torch.Generator().manual_seed(seed)
    train_size = int(0.8*len(data))
    val_size = len(data) - train_size
    train, validation = torch.utils.data.random_split(data, [train_size, val_size], generator=generator)
    return (train, validation)

def training_cycle(epochs: int, loader: DataLoader, optimizer: torch.optim.Adam, 
                   criterion: nn.CrossEntropyLoss, model: TitanicModel):

    for epoch in range(epochs):

        running_loss = 0.0

        for index, data in enumerate(loader):
            data, labels = data

            optimizer.zero_grad()

            predictions = model(data)

            loss : torch.Tensor = criterion(predictions, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f'Loss: {running_loss / len(loader):.4f}')

def validate(loader: DataLoader, model : TitanicModel):

    correct = 0.0
    total = 0.0

    model.eval()

    with torch.no_grad():

        for data in loader:
            data, labels = data

            predictions = model(data)

            _, predictions = torch.max(predictions, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
        
    accuracy = 100 * (correct / total)
    print(f'Accuracy: {accuracy}%')

def main():
    train_data = pd.read_csv("data/training_data.csv")
    train_labels = pd.read_csv("data/training_labels.csv")['Survived']

    train_data = torch.FloatTensor(train_data.values)
    train_labels = torch.LongTensor(train_labels.values)

    titanic_dataset = TensorDataset(train_data, train_labels)

    train, validation = test_train_split(titanic_dataset, 42)

    train_loader = DataLoader(train, batch_size=32)
    test_loader = DataLoader(validation, batch_size=8)

    model = TitanicModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    training_cycle(epochs=10, loader=train_loader, optimizer=optimizer, criterion=criterion, model=model)
    
    torch.save(model.state_dict(), "titanic_model.pth")

    validate(loader=test_loader, model=model)

if __name__ == "__main__":
    main()