import torch
from torch.utils.data import DataLoader
from crnn.dataset import CardDataset
from crnn.model import CRNN
from crnn.ctc_decoder import ctc_decode
from crnn.config import *


def predict(image, category):
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    if category == 'card_number':
        reload_checkpoint = './models/crnn_lcnet_card.pt'
        backbone = 'LCNet'
    elif category == 'date':
        reload_checkpoint = './models/crnn_date.pt'
        backbone = 'ResNet'

    img_width = image.size[0] * 32 // image.size[1]
    img_width = img_width // 4 * 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    crnn = CRNN(1, img_height, img_width, num_class=num_class,backbone = backbone)
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    predict_dataset = CardDataset(image_dir = image, mode="pred", img_height = img_height, img_width=img_width)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data in predict_loader:
            images = data.to(device)
            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                                label2char=LABEL2CHAR)
            print(preds)
            return preds
