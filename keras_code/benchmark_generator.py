import argparse
import os
from datetime import datetime

from tqdm import tqdm
from yaml import load

import data_loaders


def train(cli_args, log_dir):
    config = load(open(cli_args.config, "rb"))
    if config is None:
        print("Please provide a config.")

    # Load Data + Labels
    DataLoader = getattr(data_loaders, config["data_loader"])

    train_data_generator = DataLoader(config["train_data_dir"], config)

    for (data, labels) in tqdm(train_data_generator.get_data(), 'its'):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', dest='weights')
    parser.add_argument('--config', dest='config', default="config.yaml")
    cli_args = parser.parse_args()

    log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print("Logging to {}".format(log_dir))

    train(cli_args, log_dir)
