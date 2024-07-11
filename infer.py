import torch
import librosa
from scipy.io.wavfile import write

from model import UNet
from model import ISTFTModule

def load_checkpoint(model_load_path):
    checkpoint = torch.load(model_load_path)
    model = UNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def enhance_speech(model, input_path, output_path):
    istft = ISTFTModule()
    # Load the audio file
    waveform, sample_rate = librosa.load(input_path)
    print(waveform.shape)
    # Convert to torch tensor
    waveform = torch.from_numpy(waveform).float()
    
    # Ensure the audio is mono and has the correct shape
    
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    # if sample_rate != 16000:
    #     resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    #     waveform = resampler(waveform)
    
    # Normalize the waveform
    # waveform = waveform / waveform.abs().max()
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        enhanced_noise, inp_phase = model(waveform)
    
    # Reshape the output to match the input shape
    enhanced_noise = enhanced_noise.squeeze(1)
    enhanced_noise = torch.squeeze(enhanced_noise, dim=0)
    enhanced_noise = enhanced_noise.permute(1, 0)
    enhanced_noise = enhanced_noise.flatten()
    
    print(enhanced_noise.shape, waveform.shape)
    enhanced_waveform = waveform - enhanced_noise
    
    enhanced_stft = torch.tensor(enhanced_waveform).squeeze(1) * torch.exp(1j * inp_phase)
    output_ = istft(enhanced_stft)
    
    # Convert back to numpy array
    enhanced_waveform = output_.squeeze().numpy()
    
    # Save the enhanced audio
    write(output_path, 16000, (enhanced_waveform * 32767).astype('int16'))

# Load the trained model
model = load_checkpoint("trainedModel/checkpoint_epoch_10.pth")

# Enhance speech
input_file = "mix_36.wav"
output_file = "enhanced_speech.wav"
enhance_speech(model, input_file, output_file)