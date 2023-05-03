data_dir = 'D:\Softwares\Python\CreditCard-OCR\datasets/recognition/processed/card-number/attempt'
img_width = 128
img_height = 32
epochs = 100
train_batch_size = 64
lr = 0.0001
reload_checkpoint = None
early_stop = 50
eval_batch_size = 32
decode_method = 'greedy'
beam_size = 10
num_workers = 0
map_to_seq_hidden = 64
rnn_hidden = 256
leaky_relu = False
# adam / sgd / rmsprop
optim_config = 'adam'
# ResNet / LCNet / MobileNet
backbone = 'LCNet'
CHARS = '0123456789/'
num_class = len(CHARS) + 1