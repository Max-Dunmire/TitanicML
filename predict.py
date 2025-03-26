import torch
import pandas as pd
from model import TitanicModel

def main():
    test_data = pd.read_csv("data/test_model.csv")
    model = TitanicModel()
    model.load_state_dict(torch.load("titanic_model.pth"))
    t_test_data = torch.FloatTensor(test_data.values)
    predictions = model(t_test_data)
    predictions = pd.Series(torch.max(predictions, 1)[1])
    passenger_id = test_data['PassengerId']
    predictions.name = "Survived"
    combined : pd.DataFrame = pd.concat([passenger_id, predictions], axis=1)
    combined.to_csv("data/predictions.csv", index=False)

if __name__ == "__main__":
    main()