"""Usage: predict.py [-m MODEL] [-s BS] [-d DECODE] [-b BEAM] [IMAGE ...]

-h, --help    show this
-m MODEL     model file [default: ./checkpoints/crnn_synth90k.pt]
-s BS       batch size [default: 256]
-d DECODE    decode method (greedy, beam_search or prefix_beam_search) [default: beam_search]
-b BEAM   beam size [default: 10]

"""
import glob
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import common_config as config
from dataset import Synth90kDataset
from model import CRNN
from ctc_decoder import ctc_decode


def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            images = data.to(device)

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds

            pbar.update(1)
        pbar.close()

    return all_preds


def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')


def main():

    images_glob = 'D:\Softwares\Python\CreditCard-OCR\/demo/*.jpg'
    image_dir = "D:\Softwares\Python\CreditCard-OCR\demo"
    images = glob.glob(images_glob)
    file_names = os.listdir(image_dir)
    reload_checkpoint = 'crnn.pt'

    batch_size = 32
    decode_method = 'greedy'
    beam_size = 10

    img_height = 32
    img_width = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    predict_dataset = Synth90kDataset(image_dir = image_dir, file_names = file_names,mode="val",
                                      img_height=img_height, img_width=img_width)


    predict_loader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False)
    

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    preds = predict(crnn, predict_loader, Synth90kDataset.LABEL2CHAR,
                    decode_method=decode_method,
                    beam_size=beam_size)

    show_result(images, preds)


if __name__ == '__main__':
    main()
