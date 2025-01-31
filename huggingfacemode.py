import scipy
import torch
from transformers import AutoTokenizer, AutoModelForTextToWaveform

tokenizer = AutoTokenizer.from_pretrained("atul10/nepali_female_v2")
model = AutoModelForTextToWaveform.from_pretrained("atul10/nepali_female_v2")
text = "म पनि जान्छु है त अहिले लाई"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)