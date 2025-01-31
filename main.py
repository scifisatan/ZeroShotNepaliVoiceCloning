import os
import json
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter




ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs'
base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
os.makedirs(output_dir, exist_ok=True)

source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
reference_speaker = r'C:\Users\Abi\Documents\Codes\ZeroShotNepaliVoiceCloning\resources\example_reference.mp3'
target_se, audio_name = se_extractor.get_se(
    reference_speaker, tone_color_converter, target_dir='processed'
)

save_path = f'{output_dir}/output_en_default.wav'
text = "This audio is generated by OpenVoice."
src_path = f'{output_dir}/tmp.wav'
base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)
tone_color_converter.convert(
    audio_src_path=src_path,
    src_se=source_se,
    tgt_se=target_se,
    output_path=save_path,
)

print("Done generating output_en_default.wav")