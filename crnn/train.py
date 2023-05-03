import os
import csv
import torch
from config import *
from model import CRNN
import torch.optim as optim
from torch.nn import CTCLoss
from evaluate import evaluate
from torch.utils.data import DataLoader
from dataset import CardDataset, cardnumber_collate_fn

# 训练每个batch
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
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    # 加载数据集和模型
    train_dataset = CardDataset(image_dir=data_dir+'/train', mode='train',
                                    img_height=img_height, img_width=img_width)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=cardnumber_collate_fn)
    num_class = len(CardDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=map_to_seq_hidden,
                rnn_hidden=rnn_hidden,
                leaky_relu=leaky_relu,
                backbone=backbone)
    print(crnn)

    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)
    # 优化器和损失函数
    if optim_config == 'adam':
        optimizer = optim.Adam(crnn.parameters(), lr=lr)
    elif optim_config == 'sgd':
        optimizer = optim.SGD(crnn.parameters(), lr=lr)
    elif optim_config == 'rmsprop':
        optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    best_accuracy = -1
    best_epoch = None
    data = []
    # 保存路径
    if not os.path.exists('./runs/recognition'):
        os.mkdir('./runs/recognition')
    run = 1
    while os.path.exists('./runs/recognition/run'+str(run)):
        run += 1
    os.mkdir('./runs/recognition/run'+str(run))
    os.mkdir('./runs/recognition/run'+str(run)+'/checkpoints')
    save_path = './runs/recognition/run'+str(run)

    # 训练
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        total_train_loss = 0.
        total_train_count = 0
        index = 1
        length = len(train_loader)
        # 一个epoch的训练
        for train_data in train_loader: 
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)
            total_train_loss += loss
            total_train_count += train_size
            print('train_batch_loss[', index, ' / ', length, ']: ', loss / train_size, end="\r")
            index += 1
        # 保存数据
        print('total_train_loss: ', total_train_loss / total_train_count)
        temp = []
        temp.append(epoch)
        temp.append(total_train_loss / total_train_count)

        torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn last.pt')
        # print('save model at ', save_model_path)
        # 评估该epoch的结果
        test_loss, accuracy, val_loss, val_accu= evaluate(crnn, data_dir)
        temp.append(val_loss)
        temp.append(val_accu)
        temp.append(test_loss)
        temp.append(accuracy)
        data.append(temp)
        # print(wrong_cases)
        print('val_loss: ', val_loss)
        print('val_accu: ', val_accu)
        print('test_loss: ', test_loss)
        print('accuracy: ', accuracy)

        with open(save_path + '/results.csv', 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','train_loss','val_loss', 'val_accu', 'test_loss', 'accuracy'])
            writer.writerows(data)
        # 保存最好的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn best.pt')
            print('save model at ' + save_path + '/checkpoints/crnn best.pt')
        # earlystop策略
        elif epoch - best_epoch > early_stop:
            print('early stopped because not improved for {} epochs'.format(early_stop))
            break

    print('best epoch:', best_epoch)
    print('best accuracy:', best_accuracy)


if __name__ == '__main__':
    main()