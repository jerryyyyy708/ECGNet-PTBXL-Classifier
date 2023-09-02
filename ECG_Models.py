import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ECG_Data(Dataset):
    def __init__(self, x, y, mode = '1d'):
        self.x = x
        self.y = y
        self.mode = mode
    
    def __getitem__(self,index):
        x = self.x[index]
        if self.mode == '1d':
            x = x.reshape(12, 1000)
        elif self.mode == '2d':
            x = x.reshape(1, 12, 1000)
        label = self.y[index]
        if 'NORM' in label:
            y = 0
        else:
            y = 1
        return torch.from_numpy(x).float(), torch.tensor(y).float()
    
    def __len__(self):
        return len(self.y)


#kernel size -2  ->  padding -1
class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsample=None):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        #print(out.shape)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        #print(out.shape)

        if self.downsample is not None:
            out = self.maxpool(out)
            identity = self.downsample(x)

        out += identity
        # print(out.shape)

        return out




class ECGNetLite(nn.Module):

    def __init__(self, struct=[15, 17, 19, 21], in_channels=12 , fixed_kernel_size=9, num_classes=2):
        super(ECGNetLite, self).__init__()
        self.struct = struct
        self.planes = 16
        self.parallel_conv = nn.ModuleList()

        for i, kernel_size in enumerate(struct):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size,
                               stride=1, padding=0, bias=False)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_layer(kernel_size=fixed_kernel_size, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=8, stride=8, padding=2)
        self.rnn = nn.LSTM(input_size=12, hidden_size=40, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(in_features=760, out_features=num_classes)

    def _make_layer(self, kernel_size, stride, blocks=8, padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            if (i + 1) % 4 == 0:
                downsample = nn.Sequential(
                    nn.Conv1d(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=1,
                               stride=1, padding=0, bias=False),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
                self.planes += base_width
            elif (i + 1) % 2 == 0:
                downsample = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
            else:
                downsample = None
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))

        return nn.Sequential(*layers)



    def forward(self, x):
        out_sep = []

        for i in range(len(self.struct)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  # out => [b, 16, 9960]
        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  # out => [b, 64, 10]
        out = out.reshape(out.shape[0], -1)  # out => [b, 640]

        rnn_out, (rnn_h, rnn_c) = self.rnn(x.permute(2, 0, 1))
        new_rnn_h = rnn_h[-1, :, :]  # rnn_h => [b, 40]

        new_out = torch.cat([out, new_rnn_h], dim=1)  # out => [b, 680]
        result = self.fc(new_out)  # out => [b, 20]

        # print(out.shape)

        return result

class ECGNet(nn.Module):

    def __init__(self, struct=[15, 17, 19, 21], in_channels= 12, fixed_kernel_size=17, num_classes=2):
        super(ECGNet, self).__init__()
        self.struct = struct
        self.planes = 16
        self.parallel_conv = nn.ModuleList()

        for i, kernel_size in enumerate(struct):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size,
                               stride=1, padding=0, bias=False)
            self.parallel_conv.append(sep_conv)
        # self.parallel_conv.append(nn.Sequential(
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        #     nn.Conv1d(in_channels=1, out_channels=self.planes, kernel_size=1,
        #                        stride=1, padding=0, bias=False)
        # ))

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_layer(kernel_size=fixed_kernel_size, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=8, stride=8, padding=2)
        self.rnn = nn.LSTM(input_size=12, hidden_size=40, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(in_features=168, out_features=num_classes)


    def _make_layer(self, kernel_size, stride, blocks=15, padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            if (i + 1) % 4 == 0:
                downsample = nn.Sequential(
                    nn.Conv1d(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=1,
                               stride=1, padding=0, bias=False),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
                self.planes += base_width
            elif (i + 1) % 2 == 0:
                downsample = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
            else:
                downsample = None
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))

        return nn.Sequential(*layers)



    def forward(self, x):
        out_sep = []

        for i in range(len(self.struct)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  # out => [b, 16, 9960]

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  # out => [b, 64, 10]
        out = out.reshape(out.shape[0], -1)  # out => [b, 640]

        rnn_out, (rnn_h, rnn_c) = self.rnn(x.permute(2, 0, 1))
        new_rnn_h = rnn_h[-1, :, :]  # rnn_h => [b, 40]

        new_out = torch.cat([out, new_rnn_h], dim=1)  # out => [b, 680]
        result = self.fc(new_out)  # out => [b, 20]

        # print(out.shape)

        return result


class CNN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, num_classes, stride=1):
        super(CNN1D, self).__init__()
        cnn_layers = []
        for kernel_size in kernel_sizes:
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride))
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels = out_channels + 16
        self.cnn_layers = nn.Sequential(*cnn_layers)
        
        # Calculate the output size of the last convolutional layer
        final_conv_length = 925696
        self.fc = nn.Linear(4628480 , num_classes)
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        
        return x
    
    def _calculate_final_conv_length(self, input_size, kernel_sizes, stride):
        final_length = input_size
        for kernel_size in kernel_sizes:
            final_length = (final_length - kernel_size) // stride + 1
        return final_length
