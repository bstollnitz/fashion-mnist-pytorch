"""Gets the data from the cloud and saves it locally."""

import logging
import argparse

from torchvision import datasets
from torchvision.transforms import ToTensor

from common import DATA_DIR


def _save_data(data_dir: str) -> None:
    """
    Downloads Fashion MNIST data, and saves it to the path specified.
    """
    datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor(),
    )


def main() -> None:
    logging.info("Getting Fashion MNIST data from the cloud.")

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir
    logging.info("data_dir: %s", data_dir)
    _save_data(data_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
