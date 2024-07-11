import torch
import torch.nn as nn


class SpeechEnhancementLoss(nn.Module):
    def __init__(self, alpha=0.5, window_size=512, hop_length=256):
        super().__init__()
        self.alpha = alpha
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.window = torch.hann_window(window_size).to(self.device)
        self.stft = lambda x: torch.stft(x, 
                                         n_fft=window_size, 
                                         hop_length=hop_length, 
                                         win_length=window_size, 
                                         window=self.window, 
                                         return_complex=True).to(self.device)

    def si_snr(self, estimated, target):
        target = target - torch.mean(target, dim=1, keepdim=True)
        estimated = estimated - torch.mean(estimated, dim=1, keepdim=True)
        
        s_target = torch.sum(estimated * target, dim=1, keepdim=True) * target / torch.sum(target ** 2, dim=1, keepdim=True)
        e_noise = estimated - s_target
        
        si_snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / torch.sum(e_noise ** 2, dim=1))
        return -torch.mean(si_snr)

    def stft_loss(self, estimated, target):
        estimated_stft = self.stft(estimated.squeeze(1))
        target_stft = self.stft(target.squeeze(1))
        
        mag_loss = torch.mean(torch.abs(torch.abs(estimated_stft) - torch.abs(target_stft)))
        phase_loss = torch.mean(torch.abs(1 - torch.cos(torch.angle(estimated_stft) - torch.angle(target_stft))))
        
        return mag_loss + phase_loss

    def forward(self, estimated, target):
        si_snr_loss = self.si_snr(estimated, target)
        stft_loss = self.stft_loss(estimated, target)
        return self.alpha * si_snr_loss + (1 - self.alpha) * stft_loss
