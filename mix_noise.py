import os
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def adjust_noise_length(noise, target_length):
    if len(noise) < target_length:
        repeats = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeats)
    return noise[:target_length]

def mix_audio(clean, noise, snr_db):
    
    clean_power = np.sum(clean ** 2) / len(clean)
    noise_power = np.sum(noise ** 2) / len(noise)
    # print(clean_power, noise_power)
    
    #formula for snr [snr = 10log10(ps/pn)]
    desired_noise_power = clean_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(desired_noise_power / noise_power)
    
    return clean + noise


def mix_audios(clean_audio_dir, noise_audio_dir, mixed_audio_path, 
               clean_audio_path, dataset_path, 
               snr_db, sr, min_length, segment_length):
    counter = 0
    noise_audio = load_audio(noise_audio_dir)
    for root, dirs, files in tqdm(os.walk(clean_audio_dir), desc="Processing"):
        for file in files:
            if file.endswith('.flac'):
                clean_path = os.path.join(root, file)
                
                clean_audio = load_audio(clean_path)
                
                if len(clean_audio) / sr < min_length:
                    continue
                
                num_segments = len(clean_audio) // (segment_length * sr)
                # segments = [clean_audio[i * segment_length * sr : (i + 1) * segment_length * sr] for i in range(num_segments)]
                
                mixed_segments = []
                # for segment in segments:
                for i in range(num_segments):
                    segment = clean_audio[i * segment_length * sr : (i + 1) * segment_length * sr]
                    adjusted_noise = adjust_noise_length(noise_audio, len(segment))
                    mixed_audio = mix_audio(segment, adjusted_noise, snr_db)
                    
                    #save mix and clean
                    mix_segment_file_name = f"mix_{counter}.wav"
                    clean_segment_file_name = f"clean_{counter}.wav"
                    
                    os.makedirs(os.path.join(dataset_path, mixed_audio_path), exist_ok=True)
                    os.makedirs(os.path.join(dataset_path, clean_audio_path), exist_ok=True)
                    
                    mixed_output_audio_path = os.path.join(dataset_path, mixed_audio_path, mix_segment_file_name)
                    new_clean_output_audio_path = os.path.join(dataset_path, clean_audio_path, clean_segment_file_name)
                    
                    
                    sf.write(mixed_output_audio_path, mixed_audio, samplerate=sr)
                    sf.write(new_clean_output_audio_path, segment, samplerate=sr)
                    
                    mixed_segments.append(mixed_audio)
                    counter+=1
    print("Mix and Clean Audio Saved")
if __name__ == "__main__":

    sr = 16000 
    snr_db = 5
    min_length = 4
    segment_length = 4
 
    clean_audio_dir = "clean_audios"  #directory for clean audios
    noise_audio_dir = "noise.wav"
    
    mixed_audio_path = "mix"
    clean_audio_path = "clean"
    dataset_path = "dataset"
    
    mix_audios(clean_audio_dir, noise_audio_dir, mixed_audio_path, 
               clean_audio_path, dataset_path, 
               snr_db, sr, min_length, segment_length
               )
