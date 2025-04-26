import librosa
from scipy.io.wavfile import write


file_name = "one"
audio_file = f'./data/{file_name}.mp3'
y, sr = librosa.load(audio_file)

write(f'./data/{file_name}_cut.wav', sr, y[2000000:2300000])