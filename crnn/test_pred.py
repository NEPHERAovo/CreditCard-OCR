import torch
from torch.utils.data import DataLoader

from crnn.dataset import Synth90kDataset
from crnn.model import CRNN
from crnn.ctc_decoder import ctc_decode

    
# def predict(crnn, dataloader, label2char, decode_method, beam_size):


#     crnn.eval()

#     all_preds = []
#     with torch.no_grad():
#         for data in dataloader:
#             device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'

#             images = data.to(device)

#             logits = crnn(images)
#             log_probs = torch.nn.functional.log_softmax(logits, dim=2)

#             preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
#                                label2char=label2char)
#             all_preds += preds

#     return all_preds

def predict(image):
    CHARS = '0123456789'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    reload_checkpoint = 'checkpoints/crnn 69000.pt'

    batch_size = 32
    decode_method = 'greedy'
    # decode_method = 'beam_search'
    beam_size = 10
    img_height = 32
    # img_width = 128
    img_width = image.size[0] * 32 // image.size[1]
    img_width = img_width // 4 * 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    crnn = CRNN(1, img_height, img_width, 11)
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    predict_dataset = Synth90kDataset(image_dir = image, mode="pred", img_height = img_height, img_width=img_width)
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for data in predict_loader:
            images = data.to(device)
            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                                label2char=LABEL2CHAR)
            print(preds)
            return preds
    # predict_loader = DataLoader(
    #     dataset=predict_dataset,
    #     batch_size=batch_size,
    #     shuffle=False)

    # for i in predict_loader:
    #     print(i)

    # num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    # crnn = CRNN(1, img_height, img_width, num_class)
    # crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    # crnn.to(device)

    # preds = predict(crnn, predict_loader, Synth90kDataset.LABEL2CHAR,
    #                 decode_method=decode_method,
    #                 beam_size=beam_size)

    # show_result(file_names, preds)

