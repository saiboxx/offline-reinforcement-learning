import os
import gym
import yaml
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import tensor
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam
from torchsummary import summary
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage, Grayscale, Resize, ToTensor
from src.atari.utils.networks import VAE, AE, ConvEncoder, ConvDecoder, Encoder, Decoder
from src.atari.utils.data import VAEDataset


def main():
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    if cfg['AUTO_GENERATE']:
        generate(cfg)

    run(cfg)


def generate(cfg: dict):
    print('Loading environment {}.'.format(cfg['ATARI_ENV']))
    env = gym.make(cfg['ATARI_ENV'])
    env.reset()
    action_space = 3
    action_map = {0: 0, 1: 2, 2: 3}
    state_batch = []

    for _ in tqdm(range(150000)):
        action = np.asarray(random.randint(0, action_space - 1))
        new_state, _, done, _ = env.step(action_map[int(action)])
        state = preprocess_state(new_state)
        state_batch.append(state)

        if done:
            env.reset()

    state_batch = torch.stack(state_batch)
    torch.save(state_batch, cfg['AUTO_SAVE_PATH'] + '/autoencoder_data.pt')
    env.close()


def run(cfg: dict):
    print('Initializing Autoencoder.')
    input_shape = (1, 80, 80)
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    autoencoder = VAE(encoder, decoder)
    optimizer = Adam(autoencoder.parameters(), lr=0.001)
    bce_loss = BCELoss()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Utilizing device {}.'.format(device))
    encoder.to(device)
    decoder.to(device)
    autoencoder.to(device)
    summary(autoencoder, input_size=input_shape)

    print('Loading training data.')
    training_data = VAEDataset(cfg['AUTO_SAVE_PATH'])
    data_loader = DataLoader(dataset=training_data,
                             batch_size=cfg['AUTO_BATCH_SIZE'],
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True)

    print('Start training with {} epochs'.format(cfg['AUTO_EPOCHS']))
    kl_weight = 0
    for epoch in range(1, cfg['AUTO_EPOCHS'] + 1):
        for i_batch, sample in enumerate(tqdm(data_loader)):
            sample = sample.to(device)

            # Get reconstructed sample & mean / log var
            x, mu, logvar = autoencoder(sample)

            # Reconstruction Loss
            rec_loss = bce_loss(x, sample)

            # Kullback-Leibler Divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (80 * 80 * 128)

            # Total Loss
            loss = rec_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Anneal weight of KL-divergence
            kl_weight = get_kl_weight(epoch)

            if i_batch % 200 == 0:
                img_path = 'data/autoencoder/ep_{}_batch_{}.png'.format(epoch, i_batch)
                os.makedirs('data/autoencoder', exist_ok=True)
                with torch.no_grad():
                    rand_img = sample[random.randint(0, len(sample) - 1)]
                    x, _, _ = autoencoder(rand_img.unsqueeze(0))
                    boundary = torch.ones(1, 80, 1).to(device)
                    img = torch.cat([rand_img, boundary, x.squeeze(0)], 2)
                    save_image(img, img_path)

        print('Epoch {0:>3}; {1:5.4f} total loss; {2:5.4f} rec. loss; {3:5.4f} kl loss'\
              .format(epoch, loss, rec_loss, kl_loss))

    os.makedirs(cfg['AUTO_SAVE_PATH'], exist_ok=True)
    torch.save(autoencoder.state_dict(), os.path.join(cfg['AUTO_SAVE_PATH'], 'autoencoder.pt'))
    torch.save(encoder.state_dict(), os.path.join(cfg['AUTO_SAVE_PATH'], 'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join(cfg['AUTO_SAVE_PATH'], 'decoder.pt'))
    print('Saved trained models.')


def get_kl_weight(epoch: int) -> float:
    min_epoch = 3
    max_epoch = 15

    if epoch < min_epoch:
        return 0
    elif epoch >= max_epoch:
        return 1
    else:
        return (epoch - min_epoch) / (max_epoch - min_epoch)


if __name__ == '__main__':
    main()
