"""Training and evaluation."""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from common import DATA_DIR, MODEL_DIR
from neural_network import NeuralNetwork


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


def _fit(device: str, dataloader: DataLoader, model: nn.Module,
         loss_fn: CrossEntropyLoss,
         optimizer: Optimizer) -> Tuple[float, float]:
    """
    Trains the given model for a single epoch.
    """
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    # Used for printing only.
    batch_count = len(dataloader)
    print_every = 100

    model.to(device)
    model.train()

    for batch_index, (x, y) in enumerate(dataloader):
        x = x.float().to(device)
        y = y.long().to(device)

        (y_prime, loss) = _fit_one_batch(x, y, model, loss_fn, optimizer)

        correct_item_count += (y_prime.argmax(1) == y).sum().item()
        loss_sum += loss.item()
        item_count += len(x)

        # Printing progress.
        if ((batch_index + 1) % print_every == 0) or ((batch_index + 1)
                                                      == batch_count):
            accuracy = correct_item_count / item_count
            average_loss = loss_sum / item_count
            print(f"[Batch {batch_index + 1:>3d} - {item_count:>5d} items] " +
                  f"loss: {average_loss:>7f}, " +
                  f"accuracy: {accuracy*100:>0.1f}%")

    average_loss = loss_sum / item_count
    accuracy = correct_item_count / item_count

    return (average_loss, accuracy)


def _fit_one_batch(x: torch.Tensor, y: torch.Tensor, model: NeuralNetwork,
                   loss_fn: CrossEntropyLoss,
                   optimizer: Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trains a single minibatch (backpropagation algorithm).
    """
    y_prime = model(x)
    loss = loss_fn(y_prime, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (y_prime, loss)


def _evaluate(device: str, dataloader: DataLoader, model: nn.Module,
              loss_fn: CrossEntropyLoss) -> Tuple[float, float]:
    """
    Evaluates the given model for the whole dataset once.
    """
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.float().to(device)
            y = y.long().to(device)

            (y_prime, loss) = _evaluate_one_batch(x, y, model, loss_fn)

            correct_item_count += (y_prime.argmax(1) == y).sum().item()
            loss_sum += loss.item()
            item_count += len(x)

        average_loss = loss_sum / item_count
        accuracy = correct_item_count / item_count

    return (average_loss, accuracy)


def _evaluate_one_batch(
        x: torch.tensor, y: torch.tensor, model: NeuralNetwork,
        loss_fn: CrossEntropyLoss) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates a single minibatch.
    """
    with torch.no_grad():
        y_prime = model(x)
        loss = loss_fn(y_prime, y)

    return (y_prime, loss)


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
        logging.info(f"\nEpoch {epoch + 1}\n-------------------------------")
        (train_loss, train_accuracy) = _fit(device, train_dataloader, model,
                                            loss_fn, optimizer)
        logging.info(f"Train loss: {train_loss:>8f}, " +
                     f"train accuracy: {train_accuracy * 100:>0.1f}%")

    logging.info("\n***Evaluating***")
    (test_loss, test_accuracy) = _evaluate(device, test_dataloader, model,
                                           loss_fn)
    logging.info(f"Test loss: {test_loss:>8f}, " +
                 f"test accuracy: {test_accuracy * 100:>0.1f}%")

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
