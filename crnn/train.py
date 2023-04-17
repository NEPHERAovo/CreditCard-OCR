import os

import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CTCLoss

from dataset import Synth90kDataset, synth90k_collate_fn
from model import CRNN

def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    epochs = 100
    train_batch_size = 32

    lr = 0.0001
    show_interval = 10
    valid_interval = 500
    save_interval = 1000
    cpu_workers = 0
    reload_checkpoint = None

    img_width = 512
    img_height = 32
    data_dir = 'D:\Softwares\Python\CreditCard-OCR\datasets/recognition\processed'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    train_dataset = Synth90kDataset(root_dir=data_dir,image_dir=data_dir+'/train', mode='train',
                                    img_height=img_height, img_width=img_width)


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=cpu_workers,
        collate_fn=synth90k_collate_fn)


    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=64,
                rnn_hidden=256,
                leaky_relu=False)
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    assert save_interval % valid_interval == 0 or valid_interval % save_interval ==0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                print('train_batch_loss[', i, ']: ', loss / train_size)

            if i % save_interval == 0:
                save_model_path = os.path.join('D:\Softwares\Python\CreditCard-OCR/checkpoints/',"crnn "+str(i)+".pt")
                torch.save(crnn.state_dict(), save_model_path)
                print('save model at ', save_model_path)

            i += 1

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    main()