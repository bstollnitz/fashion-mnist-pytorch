"""Training and evaluation."""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from common import DATA_DIR, MODEL_DIR
from neural_network import NeuralNetwork
from utils_train_nn import fit, evaluate


def _load_data(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Returns two DataLoader objects that wrap training and test data.
    """

    training_data = datasets.FashionMNIST(data_dir,
                                          train=True,
                                          download=False,
                                          transform=ToTensor())
    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_data = datasets.FashionMNIST(data_dir,
                                      train=False,
                                      download=False,
                                      transform=ToTensor())
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return (train_dataloader, test_dataloader)


def _save_model(model_dir, model: nn.Module) -> None:
    """
    Saves the trained model.
    """
    Path(model_dir).mkdir(exist_ok=True)
    path = Path(model_dir, "weights.pth")
    torch.save(model.state_dict(), path)


def training_phase(data_dir: str, model_dir: str, device: str):
    """
    Trains the model for a number of epochs, and saves it.
    """
    learning_rate = 0.1
    batch_size = 64
    epochs = 5

    (train_dataloader, test_dataloader) = _load_data(data_dir, batch_size)

    model = NeuralNetwork()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    logging.info("\n***Training***")
    for epoch in range(epochs):
        logging.info("\nEpoch %d\n-------------------------------", epoch + 1)
        (train_loss, train_accuracy) = fit(device, train_dataloader, model,
                                           loss_fn, optimizer)
        logging.info("Train loss: %8f, train accuracy: %0.1f%%", train_loss,
                     train_accuracy * 100)

    logging.info("\n***Evaluating***")
    (test_loss, test_accuracy) = evaluate(device, test_dataloader, model,
                                          loss_fn)
    logging.info("Test loss: %8f, test accuracy: %0.1f%%", test_loss,
                 test_accuracy * 100)

    _save_model(model_dir, model)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--model_dir", dest="model_dir", default=MODEL_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir
    logging.info("data_dir: %s", data_dir)
    model_dir = args.model_dir
    logging.info("model_dir: %s", model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_phase(data_dir, model_dir, device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
