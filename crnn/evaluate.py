import torch
from config import *
from model import CRNN
from torch.nn import CTCLoss
from ctc_decoder import ctc_decode
from torch.utils.data import DataLoader
from dataset import CardDataset, cardnumber_collate_fn

def process(crnn, dataloader, criterion, device, decode_method, beam_size):
    total_count = 0
    total_loss = 0
    total_correct = 0
    for data in dataloader:
        images, targets, target_lengths = [i.to(device) for i in data]

        logits = crnn(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
        reals = targets.cpu().detach().numpy().tolist()
        target_lengths = target_lengths.cpu().detach().numpy().tolist()

        total_count += batch_size
        total_loss += loss.item()
        target_length_counter = 0
        for pred, target_length in zip(preds, target_lengths):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length
            if pred == real:
                total_correct += 1
            # else:
            #     wrong_cases.append((real, pred))
    return total_loss / total_count, total_correct / total_count

def evaluate(crnn, data_dir):
    test_dataset = CardDataset(image_dir=data_dir+'/test', mode='val',img_height=img_height, img_width=img_width)
    val_dataset = CardDataset(image_dir=data_dir+'/val', mode='val',img_height=img_height, img_width=img_width)
    test_loader = DataLoader(dataset=test_dataset,batch_size=eval_batch_size,shuffle=True,num_workers=num_workers,collate_fn=cardnumber_collate_fn)
    val_loader = DataLoader(dataset=val_dataset,batch_size=eval_batch_size,shuffle=True,num_workers=num_workers,collate_fn=cardnumber_collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'device: {device}')
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)
    crnn.eval()

    # wrong_cases = []
    with torch.no_grad():
        val_loss, accuracy = process(crnn, test_loader, criterion, device, decode_method, beam_size)
        fake_loss, fake_accu = process(crnn, val_loader, criterion, device, decode_method, beam_size)
        # print('wrong_cases: ', wrong_cases)
    return val_loss, accuracy, fake_loss, fake_accu
    

if __name__ ==  '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'D:\Softwares\Python\CreditCard-OCR\datasets/recognition\processed'
    crnn = CRNN(1, 32, 512, 11)
    crnn.load_state_dict(torch.load('./runs/recognition/run3/checkpoints/crnn best.pt', map_location=device))
    # crnn.load_state_dict(torch.load('crnn 69000.pt', map_location=device))
    crnn.to(device)
    test_loss, accuracy,val_loss,val_accu = evaluate(crnn, data_dir)
    print('test_loss: ', val_loss)
    print('accuracy: ', accuracy)
    print('val_loss: ', val_loss)
    print('val_accu: ', val_accu)