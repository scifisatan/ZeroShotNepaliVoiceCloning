import torch
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import soundfile as sf

tokenizer = AutoTokenizer.from_pretrained("atul10/nepali_female_v2")
model = AutoModelForTextToWaveform.from_pretrained("atul10/nepali_female_v2")
text = "“ट्राफिक नियमको पालना गरौँ, आफ्नो र अरूको ज्यान बचाऔँ ।” मूल नारा बोकी गत साता मात्र देशले बडो धुमधामसित ट्राफिक सप्ताह मनायो । दिनानुदिन बढिरहेको सडक दुघटनालार्इर्र्कम पार्ने उद्देश्य बोकेर । त्यस पुनित कार्यलार्इर्र्शिखरमा पुर्याउन नौ कक्षामा पढ्ने विद्यार्थी गोपालले पनि सात दिनसम्म ट्राफिक प्रहरीको झैं सेवा प्रदायक बनेर सहयोग गरेका थिए । यस सप्ताहमा अन्य विद्यालय जस्तो ऊ पढ्ने विद्यालयको पनि प्रत्यक्ष र उल्लेख्य सहभागिता थियो । "
inputs = tokenizer(text, return_tensors="pt")

def synthesize_speech(text, output_path="output.wav"):
    with torch.no_grad():
        output = model(**inputs).waveform
    waveform = output.cpu().numpy()
    sample_rate = model.config.sampling_rate
    sf.write(output_path, waveform.T, sample_rate)
    print(f"Audio saved as {output_path}")

if __name__ == "__main__":
    synthesize_speech(text)
