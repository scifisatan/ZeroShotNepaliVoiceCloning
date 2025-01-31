import os
from glob import glob
import numpy as np
from pydub import AudioSegment
from whisper_timestamped.transcribe import get_audio_tensor, get_vad_segments
import uuid


def split_audio_vad(audio_path, audio_name, target_dir, split_seconds=10.0):
    """
    - Uses a voice activity detection function (get_vad_segments) to find where speech occurs.
    - Collects these speech segments, concatenates them, and then splits them into pieces of equal duration.
    - Exports each piece as a separate WAV file.
    """
    SAMPLE_RATE = 16000
    audio_vad = get_audio_tensor(audio_path)
    segments = get_vad_segments(
        audio_vad,
        output_sample=True,
        min_speech_duration=0.1,
        min_silence_duration=1,
        method="silero",
    )
    segments = [(seg["start"], seg["end"]) for seg in segments]
    segments = [(float(s) / SAMPLE_RATE, float(e) / SAMPLE_RATE) for s, e in segments]
    print(segments)
    audio_active = AudioSegment.silent(duration=0)
    audio = AudioSegment.from_file(audio_path)

    for start_time, end_time in segments:
        audio_active += audio[int(start_time * 1000) : int(end_time * 1000)]

    audio_dur = audio_active.duration_seconds
    print(f"after vad: dur = {audio_dur}")
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)
    start_time = 0.0
    count = 0
    num_splits = int(np.round(audio_dur / split_seconds))
    assert num_splits > 0, "input audio is too short"
    interval = audio_dur / num_splits

    for i in range(num_splits):
        end_time = min(start_time + interval, audio_dur)
        if i == num_splits - 1:
            end_time = audio_dur
        output_file = f"{wavs_folder}/{audio_name}_seg{count}.wav"
        audio_seg = audio_active[int(start_time * 1000) : int(end_time * 1000)]
        audio_seg.export(output_file, format="wav")
        start_time = end_time
        count += 1
    return wavs_folder


def get_se(
    audio_path,
    vc_model,
    target_dir="processed",
):
    audio_name = (
        f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{str(uuid.uuid4())[:16]}"
    )
    se_path = os.path.join(target_dir, audio_name, "se.pth")

    wavs_folder = split_audio_vad(
        audio_path, target_dir=target_dir, audio_name=audio_name
    )

    audio_segs = glob(f"{wavs_folder}/*.wav")
    if len(audio_segs) == 0:
        raise NotImplementedError("No audio segments found!")

    return vc_model.extract_se(audio_segs, se_save_path=se_path), audio_name
