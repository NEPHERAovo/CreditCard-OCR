import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, n, seblock):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, k, s, padding=n, bias=False)]
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU6(inplace=True))
        if seblock == 1:
            layers.append(SEBlock(out_channels))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16): 
        super(SEBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, in_channels//r)
        self.linear2 = nn.Linear(in_channels//r, in_channels)
        
    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        out = out.view(-1, x.size(1), 1, 1)
        return x * out
        
class MobileNetV3(nn.Module):
    def __init__(self, mode='large', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        self.mode = mode 
        if mode == 'large':
            layers = [
                [16,   3, 1, 16, 0, 1], 
                [64,   3, 2, 64, 0, 1],
                [72,   5, 1, 72, 0, 1],
                [72,   5, 1, 120, 1, 1],
                [120, 5, 1, 120, 1, 1],
                [240, 3, 2, 240, 0, 1],
                [200, 5, 1, 184, 1, 1],
                [184, 5, 1, 184, 1, 1],
                [480, 3, 2, 480, 0, 1],
            ]       
        
        self.features = []
        in_channels = 3  
        
        for width, k, s, c, seblock, n in layers:
            out_channels = width * width_mult
            self.features.append(ConvBlock(in_channels, out_channels, k, s, n, seblock))  
            in_channels = out_channels  
        self.features = nn.Sequential(*self.features)
        
    def forward(self, x):
        x = self.features(x)
        return x  
class LCNet(nn.Module):
    def __init__(self, img_channel, img_height, img_width, leaky_relu=False):
        super(LCNet, self).__init__()
        
        self.conv1 = nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=2, stride=(2, 1), padding=0)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1))
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leaky_relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leaky_relu(x)
        return x
        
class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False, backbone = 'LCNet'):
        super(CRNN, self).__init__()

        if backbone == 'LCNet':
            self.cnn = LCNet(img_channel, img_height, img_width, leaky_relu)
            self.map_to_seq = nn.Linear(512 * (img_height // 16), map_to_seq_hidden)
        elif backbone == 'ResNet':
            self.cnn, (output_channel, output_height, output_width) = self._cnn_backbone(img_channel, img_height, img_width, leaky_relu)
            self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)
        elif backbone == 'MobileNet':
            self.cnn = MobileNetV3(mode='large', width_mult=1.0)
            self.map_to_seq = nn.Linear(512 * (img_height // 16), map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)

        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    # CRNN 原backbone
    def _cnn_backbone(self, img_channel, img_height, img_width, leaky_relu):
        assert img_height % 16 == 0
        assert img_width % 4 == 0

        # 超参设置
        channels = [img_channel, 64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=True):
            # shape of input: (batch, input_channel, height, width)
            input_channel = channels[i]
            output_channel = channels[i+1]

            cnn.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, kernel_sizes[i], strides[i], paddings[i])
            )

            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn.add_module(f'relu{i}', relu)

        # size of image: (channel, height, width) = (img_channel, img_height, img_width)
        conv_relu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)

        conv_relu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)

        conv_relu(2)
        conv_relu(3)
        cnn.add_module(
            'pooling2',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (256, img_height // 8, img_width // 4)

        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module(
            'pooling3',
            nn.MaxPool2d(kernel_size=(2, 1))
        )  # (512, img_height // 16, img_width // 4)

        conv_relu(6)  # (512, img_height // 16 - 1, img_width // 4 - 1)

        output_channel, output_height, output_width = \
            channels[-1], img_height // 16 - 1, img_width // 4 - 1
        return cnn, (output_channel, output_height, output_width)

    def forward(self, images):
        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)

        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output