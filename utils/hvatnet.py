import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from simple_parsing import Serializable
from dataclasses import dataclass
from typing import List, Tuple
# from torchvision import models
import numpy as np
import pywt

"""
Model
1. Input data: 200 fps 
2. Conv1d 1x1 -> N Res Blocks.
3. N + M downsample blocks with advanced Conv 
4. M upsample blocks with skip connections
5. Prediction: 25 fps
"""

@dataclass
class Config(Serializable):
    n_electrodes: int
    n_channels_out: int
    n_res_blocks: int
    n_blocks_per_layer: int
    n_filters: int
    kernel_size: int
    dilation: int
    strides: List[int]
    small_strides: List[int]
    dropout_rate: float = 0.5 # 0.2754191136439675 # 0.5

class TuneModule(nn.Module):
    def __init__(self, n_electrodes=8, temperature=5):
        super(TuneModule, self).__init__()
        """
        - interpolate signal spatially 
        - change amplitude of the signal

        n_electrodes: number of electrodes (default: 8)
        temperture: temperature for softmax of weights (default: 5)
        """
        # spatial rotation.
        self.spatial_weights = torch.nn.Parameter(torch.eye(n_electrodes, n_electrodes), requires_grad=True)
        self.temp = torch.tensor(temperature, requires_grad=False)

        # normalization + amplitude scaling
        self.layer_norm = nn.LayerNorm(n_electrodes, elementwise_affine=True, eps=1e-5)
        
    def forward(self, x):
        """
        x: batch, channel, time
        """

        x = x.permute(0, 2, 1) # batch, time, channel

        # spatial rotation
        weights = torch.softmax(self.spatial_weights*self.temp , dim=0)
        x = torch.matmul(x, weights) # batch, time, channel

        # normalization + amplitude scaling
        x = self.layer_norm(x)

        x = x.permute(0, 2, 1) # batch, channel, time

        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,dropout_rate=0.5):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

        # get number of parameters
        self.n_params = 0
        for p in self.parameters():
            self.n_params += p.numel()
        print('Number of parameters: ', self.n_params)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out
    
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', bias=True):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
#         self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
    
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         return x    

class SimpleResBlock(nn.Module):
    """
    Input is [batch, emb, time]
    Res block.
    In features input and output the same.
    So we can apply this block several times.
    """
    def __init__(self, in_channels, kernel_size, dropout_rate=0.5):
        super(SimpleResBlock, self).__init__()

        # self.conv1 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=kernel_size)
        self.conv1 = nn.Conv1d(in_channels, in_channels,
                               kernel_size=kernel_size,
                               bias=True,
                               padding='same')
        
        # self.batch_norm1 = nn.BatchNorm1d(in_channels)  
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # self.conv2 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels, in_channels,
                               kernel_size=kernel_size,
                               bias=True,
                               padding='same')

        # self.batch_norm2 = nn.BatchNorm1d(in_channels)


        ## transfer learning way ------------------------------------------------------
        # self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding='same')
        # self.bn1 = nn.BatchNorm1d(in_channels)
        # self.activation = nn.GELU()
        # self.dropout = nn.Dropout(dropout_rate)
        # self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding='same')
        # self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x_input):

        x = self.conv1(x_input)
        # x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        # x = self.batch_norm2(x)

        res = x + x_input

        return res

    ## transfer learning way ------------------------------------------------------
        # x = self.conv1(x_input)
        # x = self.bn1(x)
        # x = self.activation(x)
        # x = self.dropout(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # res = x + x_input
        # return res

class AdvancedConvBlock(nn.Module):
    """
    Input is [batch, emb, time]
    block [ conv -> layer norm -> act -> dropout ]

    To do:
        add res blocks.
    """
    def __init__(self, in_channels, kernel_size,dilation=1, dropout_rate=0.5):
        super(AdvancedConvBlock, self).__init__()

        # use it instead stride.

        self.conv_dilated = nn.Conv1d(in_channels, in_channels,
                                      kernel_size=kernel_size,
                                      dilation = dilation,
                                      bias=True,
                                      padding='same')
        # self.batch_norm_dilated = nn.BatchNorm1d(in_channels)

        self.conv1_1 = nn.Conv1d(in_channels, in_channels,
                                 kernel_size=kernel_size,
                                 bias=True,
                                 padding='same')

        self.conv1_2 = nn.Conv1d(in_channels, in_channels,
                                 kernel_size=kernel_size,
                                 bias=True,
                                 padding='same')

        # self.conv_dilated = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation)
        # self.conv1_1 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=kernel_size)
        # self.batch_norm1_1 = nn.BatchNorm1d(in_channels)
        # self.conv1_2 = DepthwiseSeparableConv(in_channels, in_channels, kernel_size=kernel_size)
        # self.batch_norm1_2 = nn.BatchNorm1d(in_channels)
       
        self.conv_final = nn.Conv1d(in_channels, in_channels,
                                    kernel_size=1,
                                    bias=True,
                                    padding='same')
        self.dropout = nn.Dropout(dropout_rate)


        ## transfer learning way ------------------------------------------------------
        # self.conv_dilated = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, dilation=dilation, padding='same')
        # self.bn_dilated = nn.BatchNorm1d(in_channels)
        # self.conv1_1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding='same')
        # self.bn1_1 = nn.BatchNorm1d(in_channels)
        # self.conv1_2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding='same')
        # self.bn1_2 = nn.BatchNorm1d(in_channels)
        # self.conv_final = nn.Conv1d(in_channels, in_channels, kernel_size=1, padding='same')
        # self.bn_final = nn.BatchNorm1d(in_channels)
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_input):
        """
        input
            - dilation
            - gated convolution
            - conv final
            - maybe dropout. and LN
        - input + res
        """
        x = self.conv_dilated(x_input)
        # x = self.batch_norm_dilated(x)

        flow = torch.tanh(self.conv1_1(x))
        # flow = self.batch_norm1_1(flow)
        gate = torch.sigmoid(self.conv1_2(x))
        # gate = self.batch_norm1_2(gate)

        res = flow * gate

        res = self.conv_final(res)
        res = self.dropout(res)
        
        res = res + x_input
        return res
    
    ## transfer learning way ------------------------------------------------------
        # x = self.conv_dilated(x_input)
        # x = self.bn_dilated(x)
        # flow = torch.tanh(self.conv1_1(x))
        # flow = self.bn1_1(flow)
        # gate = torch.sigmoid(self.conv1_2(x))
        # gate = self.bn1_2(gate)
        # res = flow * gate
        # res = self.conv_final(res)
        # res = self.bn_final(res)
        # res = self.dropout(res)
        # res = res + x_input
        # return res

class AdvancedEncoder(nn.Module):
    '''add lstm layers'''
    def __init__(self, n_blocks_per_layer=3, n_filters=64, kernel_size=3,
                 dilation=1, strides = (2, 2, 2), dropout_rate=0.5):
        super(AdvancedEncoder, self).__init__()

        self.n_layers = len(strides)
        self.downsample_blocks = nn.ModuleList([nn.Conv1d(n_filters, n_filters, 
                                                          kernel_size=stride, stride=stride) for stride in strides])

        conv_layers = []

        for i in range(self.n_layers):
            blocks = nn.ModuleList([AdvancedConvBlock(n_filters,kernel_size,
                                                      dilation=dilation, dropout_rate=dropout_rate) for i in range(n_blocks_per_layer)])
            
            layer = nn.Sequential(*blocks)
            conv_layers.append(layer)

            
        self.conv_layers = nn.ModuleList(conv_layers)


        # # # Transfer learning way ------------------------------------------------------
        # # Load a pre-trained ResNet model
        # resnet = models.resnet18(pretrained=True)
        # self.resnet_layers = list(resnet.children())[:-2]  # Remove the fully connected layer and the average pooling layer
    
        # # Replace the first convolutional layer to match the number of input channels
        # self.resnet_layers[0] = nn.Conv2d(1, n_filters, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.resnet_bn = nn.BatchNorm2d(n_filters)
        
        # # Convert the ResNet layers into a sequential model
        # self.resnet = nn.Sequential(*self.resnet_layers)
        
        # self.conv_layers = nn.ModuleList([AdvancedConvBlock(n_filters, kernel_size, dilation=dilation, dropout_rate=dropout_rate) for _ in range(n_blocks_per_layer)])
        # self.downsample_blocks = nn.ModuleList([nn.Conv1d(n_filters, n_filters, kernel_size=stride, stride=stride) for stride in strides])

     

        # # LSTM way ------------------------------------------------------

        # self.n_layers = len(strides)
        # self.downsample_blocks = nn.ModuleList([nn.Conv1d(n_filters, n_filters, 
        #                                                   kernel_size=stride, stride=stride) for stride in strides])

        # conv_layers = []
        # lstm_layers = []
        # for i in range(self.n_layers):
        #     blocks = nn.ModuleList([AdvancedConvBlock(n_filters,kernel_size,
        #                                               dilation=dilation, dropout_rate=dropout_rate) for i in range(n_blocks_per_layer)])
            
        #     layer = nn.Sequential(*blocks)
        #     conv_layers.append(layer)

        #     lstm_layer = nn.LSTM(n_filters, n_filters, batch_first=True, bidirectional=True)
        #     lstm_layers.append(lstm_layer)
            
        # self.conv_layers = nn.ModuleList(conv_layers)
        # self.lstm_layers = nn.ModuleList(lstm_layers)



    def forward(self, x):
        """
        Apply conv + downamsple
        Return uutputs of eahc conv + the last features after downsampling.
        """

        outputs =  []
        for conv_block, down in zip(self.conv_layers, self.downsample_blocks) :
            x_res = conv_block(x)
            x = down(x_res)
            outputs.append(x_res)

        outputs.append(x)

        return outputs
    
        ## Transfer learning way ------------------------------------------------------
        # print('x:', x.shape)
        # x = x.unsqueeze(1)  # Add a channel dimension
        # print('x:', x.shape)
        # x = self.resnet(x)
        # print('x:', x.shape)
        # x = self.resnet_bn(x)
        # print('x:', x.shape)
        # x = x.squeeze(2)  # Remove the height dimension
        # outputs = []
        # for conv_block, down in zip(self.conv_layers, self.downsample_blocks):
        #     x_res = conv_block(x)
        #     x = down(x_res)
        #     outputs.append(x_res)
        # outputs.append(x)
        # return outputs  

        # # LSTM way
        # outputs =  []
        # for conv_block, lstm, down in zip(self.conv_layers, self.lstm_layers, self.downsample_blocks) :

        #     x_res = conv_block(x)
        #     outputs.append(x_res)
        #     x = down(x_res)

        #     # Reshape x for LSTM: (batch_size, channels, seq_length) -> (batch_size, seq_length, channels)
        #     x = x.permute(0, 2, 1)
        #     x, _ = lstm(x)
        #     # Reshape x back: (batch_size, seq_length, channels) -> (batch_size, channels, seq_length)
        #     x = x.permute(0, 2, 1)

        # return outputs, x



class AdvancedDecoder(nn.Module):
    def __init__(self, n_blocks_per_layer=3, n_filters=64, kernel_size=3,
                 dilation=1, strides = (2, 2, 2), dropout_rate=0.5):
        super(AdvancedDecoder, self).__init__()

        self.n_layers = len(strides)

        self.upsample_blocks = nn.ModuleList([nn.Upsample(scale_factor=scale,
                                                          mode='linear',
                                                          align_corners=False) for scale in strides])

        conv_layers = []
        for i in range(self.n_layers):
           
            reduce  = nn.Conv1d(n_filters*2, n_filters, kernel_size=kernel_size, padding='same')
            conv_blocks = nn.ModuleList([AdvancedConvBlock(n_filters, kernel_size, dilation=dilation,dropout_rate=dropout_rate) for i in range(n_blocks_per_layer)])
            
            conv_blocks.insert(0, reduce)
            layer = nn.Sequential(*conv_blocks)

            conv_layers.append(layer)
        
        self.conv_layers = nn.ModuleList(conv_layers)

        # # Transfer learning way ------------------------------------------------------
        # self.n_layers = len(strides)
        # self.upsample_blocks = nn.ModuleList([nn.Upsample(scale_factor=scale, mode='linear', align_corners=False) for scale in strides])
        # self.conv_layers = nn.ModuleList()
        
        # for i in range(self.n_layers):
        #     reduce = nn.Conv1d(n_filters * 2, n_filters, kernel_size=kernel_size, padding='same')
        #     conv_blocks = nn.ModuleList([AdvancedConvBlock(n_filters, kernel_size, dilation=dilation, dropout_rate=dropout_rate) for _ in range(n_blocks_per_layer)])
        #     conv_blocks.insert(0, reduce)
        #     layer = nn.Sequential(*conv_blocks)
        #     self.conv_layers.append(layer)


        # # LSTM way ------------------------------------------------------
        
        # self.n_layers = len(strides)
        # self.upsample_blocks = nn.ModuleList([nn.ConvTranspose1d(n_filters, n_filters, 
        #                                                          kernel_size=stride, stride=stride) for stride in strides])

        # conv_layers = []
        # lstm_layers = []
        # for i in range(self.n_layers):
        #     blocks = nn.ModuleList([AdvancedConvBlock(n_filters,kernel_size,
        #                                               dilation=dilation, dropout_rate=dropout_rate) for i in range(n_blocks_per_layer)])
        #     layer = nn.Sequential(*blocks)
        #     conv_layers.append(layer)

        #     lstm_layer = nn.LSTM(n_filters, n_filters, batch_first=True, bidirectional=True)
        #     lstm_layers.append(lstm_layer)
            
        # self.conv_layers = nn.ModuleList(conv_layers)
        # self.lstm_layers = nn.ModuleList(lstm_layers)



    def forward(self, skips):
        """
        Apply conv + downamsple
        Return uutputs of each conv + the last features after downsampling.
        """
        skips = skips[::-1]
        x = skips[0]
        outputs = []
        for idx, (conv_block, up) in enumerate(zip(self.conv_layers, self.upsample_blocks)):
            x = up(x)
            x = torch.cat([x, skips[idx + 1]], 1)
            x = conv_block(x)
            outputs.append(x)
        return outputs


        # # LSTM way
        # for conv_block, lstm, up, encoder_output in zip(self.conv_layers, self.lstm_layers, self.upsample_blocks, encoder_outputs):
        #     x = up(x)
        #     x = x + encoder_output

        #     # Reshape x for LSTM: (batch_size, channels, seq_length) -> (batch_size, seq_length, channels)
        #     x = x.permute(0, 2, 1)
        #     x, _ = lstm(x)
        #     # Reshape x back: (batch_size, seq_length, channels) -> (batch_size, channels, seq_length)
        #     x = x.permute(0, 2, 1)

        #     x = conv_block(x)
            
        # return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, T = x.size()
        query = self.query_conv(x).view(batch_size, -1, T).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, T)
        value = self.value_conv(x).view(batch_size, -1, T)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, T)
        
        out = self.gamma * out + x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, T = x.size()
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x).view(batch_size, C, -1))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x).view(batch_size, C, -1))))
        out = avg_out + max_out
        return torch.sigmoid(out).expand_as(x) * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(concat)
        return self.sigmoid(out) * x

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x



class SWPTLayer(nn.Module):
    def __init__(self, wavelet='db1', level=3):
        super(SWPTLayer, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        # Assuming x is of shape (batch_size, n_channels, seq_len)
        batch_size, n_channels, seq_len = x.size()
        transformed = []
        for i in range(batch_size):
            transformed_batch = []
            for j in range(n_channels):
                # signal = x[i, j].cpu().numpy()
                signal = x[i, j].detach().cpu().numpy()
                swpt = pywt.WaveletPacket(data=signal, wavelet=self.wavelet, mode='symmetric', maxlevel=self.level)
                nodes = swpt.get_level(self.level, order='natural')
                coefficients = np.array([node.data for node in nodes]).flatten()
                transformed_batch.append(coefficients)
            transformed.append(np.stack(transformed_batch, axis=0))
        transformed = np.stack(transformed, axis=0)
        return torch.tensor(transformed, dtype=x.dtype, device=x.device)

class HVATNetv3(nn.Module):
    config = Config
    def __init__(self, config: Config):
        super(HVATNetv3, self).__init__()

        # Use the configuration from the dataclass
        self.n_inp_features = config.n_electrodes
        self.n_channels_out = config.n_channels_out
        self.model_depth = len(config.strides)

        self.tune_module = TuneModule(n_electrodes=config.n_electrodes, temperature=5.0)

        # Change number of features to custom one
        self.spatial_reduce = nn.Conv1d(config.n_electrodes, config.n_filters, kernel_size=1, padding='same')

        self.denoiser = nn.Sequential(*[SimpleResBlock(config.n_filters, config.kernel_size, config.dropout_rate) for _ in range(config.n_res_blocks)])

        self.encoder = AdvancedEncoder(n_blocks_per_layer=config.n_blocks_per_layer,
                                       n_filters=config.n_filters, kernel_size=config.kernel_size,
                                       dilation=config.dilation, strides=config.strides)

        self.mapper = nn.Sequential(nn.Conv1d(config.n_filters, config.n_filters, config.kernel_size, padding='same'), 
                                    nn.GELU(), 
                                    nn.Conv1d(config.n_filters, config.n_filters, config.kernel_size, padding='same'), 
                                    nn.GELU())

        self.encoder_small = AdvancedEncoder(n_blocks_per_layer=config.n_blocks_per_layer,
                                             n_filters=config.n_filters, kernel_size=config.kernel_size,
                                             dilation=config.dilation, strides=config.small_strides,dropout_rate=config.dropout_rate)

        self.decoder_small = AdvancedDecoder(n_blocks_per_layer=config.n_blocks_per_layer,
                                             n_filters=config.n_filters, kernel_size=config.kernel_size,
                                             dilation=config.dilation, strides=config.small_strides[::-1],dropout_rate=config.dropout_rate)
        

        # Adding the dense layer
        # self.dense = nn.Linear(config.n_filters, config.n_filters)  # Dense layer

        # self.rnn = RNN(input_size=config.n_filters, hidden_size=config.n_filters, num_layers=5, output_size=config.n_filters, dropout_rate=config.dropout_rate)
        self.simple_pred_head = nn.Conv1d(config.n_filters, config.n_channels_out, kernel_size=1, padding='same')

        # Get number of parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print('Number of parameters:', self.n_params)

        self.attention_block = AttentionBlock(config.n_filters)
        self.cbam_block = CBAMBlock(config.n_filters)  # Use CBAM here

        # self.swpt = SWPTLayer(level=3)


        ## Transfer learning way ------------------------------------------------------
        # self.n_inp_features = config.n_electrodes
        # self.n_channels_out = config.n_channels_out
        # self.model_depth = len(config.strides)
        # self.tune_module = TuneModule(n_electrodes=config.n_electrodes, temperature=5.0)
        # self.spatial_reduce = nn.Conv1d(config.n_electrodes, 1, kernel_size=1, padding='same')
        # self.denoiser = SimpleResBlock(in_channels=1, kernel_size=config.kernel_size)
        # self.encoder = AdvancedEncoder(
        #     n_blocks_per_layer=config.n_blocks_per_layer,
        #     n_filters=config.n_filters,
        #     kernel_size=config.kernel_size,
        #     dilation=config.dilation,
        #     strides=config.strides,
        #     dropout_rate=config.dropout_rate
        # )
        # self.decoder = AdvancedDecoder(
        #     n_blocks_per_layer=config.n_blocks_per_layer,
        #     n_filters=config.n_filters,
        #     kernel_size=config.kernel_size,
        #     dilation=config.dilation,
        #     strides=config.strides,
        #     dropout_rate=config.dropout_rate
        # )
        # self.downreduce = nn.Conv1d(config.n_filters * 2, config.n_filters, kernel_size=1, padding='same')
        # self.small_layers = nn.ModuleList([SimpleResBlock(config.n_filters, config.kernel_size, dropout_rate=config.dropout_rate) for _ in range(config.n_res_blocks)])
        # self.downsample_blocks = nn.ModuleList([nn.Conv1d(config.n_filters, config.n_filters, kernel_size=stride, stride=stride) for stride in config.small_strides])
        # self.attention_layer = AttentionBlock(in_channels=config.n_filters)
        # self.output_layer = nn.Linear(config.n_filters, config.n_channels_out)
        # self.layer_norm = nn.LayerNorm(config.n_filters, elementwise_affine=True, eps=1e-5)
        
        

    def forward(self, x, targets=None):
        """
        x: [batch, n_electrodes, time]
        targets: loss calculation with the same shape as x.
    
        """
        # tune inputs to model
        x = self.tune_module(x)
        # x = self.swpt(x)
        # denoising part
        x = self.spatial_reduce(x)
        x = self.denoiser(x)
        #x = self.cbam_block(x)  # Apply CBAM after denoising
        x = self.attention_block(x)  # Apply attention after denoising

        # extract features
        # TODO: add mapper and change encoder to return all features
        outputs = self.encoder(x)
        # outputs, x = self.encoder(x)
        emg_features = outputs[-1] # 25 fps features
        
        # decode features
        # 1. simple way:  mapper + pred_head + quat conversion
        # emg_features = self.mapper(emg_features)
        
        # 2. Unet way:  encoder + mapper + decoder + quat conversion
        outputs_small = self.encoder_small(emg_features)
        outputs_small[-1] = self.mapper(outputs_small[-1])
        emg_features = self.decoder_small(outputs_small)[-1]


        ## Transfer learning way ------------------------------------------------------
        # x = self.tune_module(x)
        # x = self.spatial_reduce(x)
        # x = self.denoiser(x)
        # print('x:', x.shape)
        # skips = self.encoder(x)
        # out = self.decoder(skips)
        # out = self.downreduce(out[-1])
        # for down, small in zip(self.downsample_blocks, self.small_layers):
        #     out = down(out)
        #     out = small(out)
        # out = self.layer_norm(out.permute(0, 2, 1)).permute(0, 2, 1)
        # out = self.attention_layer(out)
        # out = torch.mean(out, dim=-1)
        # emg_features = self.output_layer(out)
        # print('emg_features', emg_features.shape)


        # # 2.5 LSTM way --------------------------------------------------------------------------------------
        # # Encoder
        # # encoder_outputs, x = self.encoder(x)
        # outputs_small, x = self.encoder(emg_features)
        # outputs_small[-1] = self.mapper(outputs_small[-1])
        # # Decoder
        # # x = self.decoder(x, encoder_outputs)
        # emg_features = self.decoder_small(x, outputs_small)[-1]

        # # Output
        # x = self.output(x)
        # x = self.pred_activation(x)

        # Adding dense layer in forward pass---------------------------------------------------------------
        # emg_features = emg_features.permute(0, 2, 1)  # (batch, time, channels) -> (batch, channels, time)
        # emg_features = self.dense(emg_features)  # Apply the dense layer
        # emg_features = emg_features.permute(0, 2, 1)  # (batch, channels, time) -> (batch, time, channels)

        # # 3. RNN way --------------------------------------------------------------------------------------
        # emg_features  = emg_features.permute(0, 2, 1) # size [batch, n_filters, time]
        # res = self.rnn(emg_features)
        # emg_features = res.permute(0, 2, 1)


        pred = self.simple_pred_head(emg_features)

        if targets is None:
            return pred
        
        # loss = F.l1_loss(pred, targets)
        loss = F.mse_loss(pred, targets)
        # loss = F.smooth_l1_loss(pred, targets) # compared to the other losses, this one has the worset result. 

        return loss, pred

    def _to_quats_shape(self, x):
        batch, n_outs, time = x.shape
        x = x.reshape(batch, -1, 4, time)
        
        if self.training: 
            return x 
        else: 
            return F.normalize(x, p=2.0, dim=2)
        
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def inference(self, myo):
        """
        Params:
            myo: is numpy array with shape (time, n_electrodes)
        Return
            numpy array with shape (N_timestamps, 20)
        """
        self.eval()

        x = torch.from_numpy(myo)

        t, c = x.shape
        x = rearrange(x, 't c -> 1 c t', t=t, c=c)
        x = x.to(self.device).to(self.dtype)
        
        y_pred = self(x, targets=None)
        y_pred = y_pred[0].to('cpu').detach().numpy()

        return y_pred.T


# start python code 
if __name__ == '__main__':
    
    hvatnet_v3_params =dict(n_electrodes=8, n_channels_out=64,
                            n_res_blocks=3, n_blocks_per_layer=3,
                            n_filters=128, kernel_size=3,
                            strides=(2, 2, 2), dilation=2, 
                            small_strides = (2, 2))
    model = HVATNetv3(**hvatnet_v3_params)


    x = torch.randn(1, 8, 256)
    y = model(x)

    print('Input shape: ', x.shape)
    print('Output shape: ', y.shape)