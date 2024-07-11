import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self,).__init__()
        self.size_filter_in = 64
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.encoder1 = self.conv_block(self.in_channels, self.size_filter_in * 1)
        self.encoder2 = self.conv_block(self.size_filter_in * 1, self.size_filter_in * 2)
        self.encoder3 = self.conv_block(self.size_filter_in * 2, self.size_filter_in * 4)
        self.encoder4 = self.conv_block(self.size_filter_in * 4, self.size_filter_in * 8)
        
        self.bridge = self.conv_block(self.size_filter_in * 8, self.size_filter_in * 16)
        
        self.decoder4 = self.up_conv_block(self.size_filter_in * 16, self.size_filter_in * 8)
        self.decoder3 = self.up_conv_block(self.size_filter_in * 16, self.size_filter_in * 4)
        self.decoder2 = self.up_conv_block(self.size_filter_in * 8, self.size_filter_in * 2)
        self.decoder1 = self.up_conv_block(self.size_filter_in * 4, self.size_filter_in)
        
        self.final_conv = nn.Conv2d(self.size_filter_in * 2, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        self._initialize_weights()
        
        
        # self.stft = STFTModule()
        # self.istft = ISTFTModule()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
    
    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        
    def pad_to_match(self, x, y):
        diff_h = y.size(2) - x.size(2)
        diff_w = y.size(3) - x.size(3)
        
        if diff_h > 0 or diff_w > 0:
            x = torch.nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        return x
    
    def forward(self, x):
        #stft
        # x_mag, x_phase = self.stft(x)
        x = x.unsqueeze(1)
        
        #encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(self.dropout(e3)))

        bridge = self.bridge(self.pool(e4))
        # drop5 = self.dropout(bridge)
        # print(bridge.shape)
        # Decoder
        d4 = self.decoder4(bridge)
        d4 = self.pad_to_match(d4, e4)
        
        d3 = self.decoder3(torch.cat([d4, e4], dim=1))
        
        d2 = self.decoder2(torch.cat([d3, e3], dim=1))
        d2 = self.pad_to_match(d2, e2)
        
        d1 = self.decoder1(torch.cat([d2, e2], dim=1))
        d1 = self.pad_to_match(d1, e1)
        
        out = self.final_conv(torch.cat([d1, e1], dim=1))
        out = torch.tanh(out)
        # out = torch.nn.functional.interpolate(out, size=x_mag.shape[-2:], mode='bilinear', align_corners=False)
        # print(out.shape)
        # enhanced_stft = out.squeeze(1) * torch.exp(1j * x_phase)
        # output_ = self.istft(enhanced_stft)
        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
                    
class SpeechEnhacementModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SpeechEnhacementModel, self).__init__()
        self.unet = UNet(in_channels, out_channels)
        self.stft = STFTModule()
        # self.istft = ISTFTModule()
    
    def forward(self, x):
        x, x_phase = self.stft(x)
        out_ = self.unet(x)
        return out_, x_phase
# Huber loss function
# class HuberLoss(nn.Module):
#     def __init__(self, delta=1.0):
#         super().__init__()
#         self.delta = delta

#     def forward(self, pred, target):
#         abs_error = torch.abs(pred - target)
#         quadratic = torch.min(abs_error, torch.tensor(self.delta).to(abs_error.device))
#         linear = abs_error - quadratic
#         loss = 0.5 * quadratic**2 + self.delta * linear
#         return loss.mean()

# Example usage:
# model = UNet()
# criterion = HuberLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

class STFTModule(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        #no any spectrum breakage.
        self.window = nn.Parameter(torch.hann_window(win_length), requires_grad=False)

    def forward(self, x):
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, 
                          win_length=self.win_length, window=self.window, return_complex=True)
        return stft.abs(), stft.angle()

class ISTFTModule(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = nn.Parameter(torch.hann_window(win_length), requires_grad=False)

    def forward(self, X):
        return torch.istft(X, n_fft=self.n_fft, hop_length=self.hop_length, 
                           win_length=self.win_length, window=self.window)
       
       
       
       



# class UNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super(UNet, self,).__init__()
#         self.size_filter_in = 64
#         self.in_channels = in_channels
#         self.out_channels = out_channels
        
#         self.encoder1 = self.conv_block(self.in_channels, self.size_filter_in * 1)
#         self.encoder2 = self.conv_block(self.size_filter_in * 1, self.size_filter_in * 2)
#         self.encoder3 = self.conv_block(self.size_filter_in * 2, self.size_filter_in * 4)
#         self.encoder4 = self.conv_block(self.size_filter_in * 4, self.size_filter_in * 8)
        
#         self.bridge = self.conv_block(self.size_filter_in * 8, self.size_filter_in * 16)
        
#         self.decoder4 = self.up_conv_block(self.size_filter_in * 16, self.size_filter_in * 8)
#         self.decoder3 = self.up_conv_block(self.size_filter_in * 16, self.size_filter_in * 4)
#         self.decoder2 = self.up_conv_block(self.size_filter_in * 8, self.size_filter_in * 2)
#         self.decoder1 = self.up_conv_block(self.size_filter_in * 4, self.size_filter_in)
        
#         self.final_conv = nn.Conv2d(self.size_filter_in * 2, out_channels, kernel_size=1)
        
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout = nn.Dropout(0.3)
        
#         self._initialize_weights()
        
        
#         self.stft = STFTModule()
#         self.istft = ISTFTModule()

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(inplace=True)
#         )
    
#     def up_conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.LeakyReLU(inplace=True)
#         )
        
#     def pad_to_match(self, x, y):
#         diff_h = y.size(2) - x.size(2)
#         diff_w = y.size(3) - x.size(3)
        
#         if diff_h > 0 or diff_w > 0:
#             x = torch.nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
#                           diff_h // 2, diff_h - diff_h // 2])
#         return x
    
#     def forward(self, x):
#         #stft
#         x_mag, x_phase = self.stft(x)
#         x = x_mag.unsqueeze(1)
        
#         #encoder
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(self.pool(e1))
#         e3 = self.encoder3(self.pool(e2))
#         e4 = self.encoder4(self.pool(self.dropout(e3)))

#         bridge = self.bridge(self.pool(e4))
#         # drop5 = self.dropout(bridge)
#         # print(bridge.shape)
#         # Decoder
#         d4 = self.decoder4(bridge)
#         d4 = self.pad_to_match(d4, e4)
        
#         d3 = self.decoder3(torch.cat([d4, e4], dim=1))
        
#         d2 = self.decoder2(torch.cat([d3, e3], dim=1))
#         d2 = self.pad_to_match(d2, e2)
        
#         d1 = self.decoder1(torch.cat([d2, e2], dim=1))
#         d1 = self.pad_to_match(d1, e1)
        
#         out = self.final_conv(torch.cat([d1, e1], dim=1))
#         out = torch.tanh(out)
#         # print(out.shape)
#         # enhanced_stft = out.squeeze(1) * torch.exp(1j * x_phase)
#         # output_ = self.istft(enhanced_stft)
            
#         return out, x_phase

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
