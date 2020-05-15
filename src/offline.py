import yaml
import torch
from torch.utils.data import DataLoader
from src.utils.data import EnvironmentDataset


def main():
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    train(cfg)


def train(cfg: dict):
    print('Initializing Dataloader.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Utilizing device {}'.format(device))
    training_data = EnvironmentDataset(cfg['TRAIN_DATA_PATH'])
    data_loader = DataLoader(dataset=training_data,
                             batch_size=cfg['BATCH_SIZE'],
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True)

    print('Start training with {} epochs'.format(cfg['EPOCHS']))
    for e in range(1, cfg['EPOCHS'] + 1):
        for i_batch, sample_batched in enumerate(data_loader):
            print(sample_batched['state'].shape)
            exit()


if __name__ == '__main__':
    main()