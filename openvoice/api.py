import torch
import soundfile
from openvoice.models import VoiceConverter
import json
import os
import librosa
from openvoice.mel_processing import spectrogram_torch


class ToneColorConverter(object):
    def __init__(self, config_path: str, device: str = "cuda:0"):
        if "cuda" in device:
            assert torch.cuda.is_available()
        self.hps = self._load_config(config_path)
        self.device = device
        self.model = VoiceConverter(
            len(getattr(self.hps, "symbols", [])),
            self.hps.data.filter_length // 2 + 1,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        ).to(device)
        self.model.eval()
        self.version = getattr(self.hps, "_version_", "v1")

    def _load_config(self, config_path: str) -> dict:
        """Load JSON config file and return as dictionary."""
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_ckpt(self, ckpt_path: str):
        """Load model checkpoint."""
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict["model"], strict=False)
        print("Loaded checkpoint '{}'".format(ckpt_path))
        print("missing/unexpected keys:", a, b)

    def _prepare_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute spectrogram from audio tensor."""
        return spectrogram_torch(
            audio,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        ).to(self.device)

    def extract_se(self, ref_wav_list, se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]

        gs = []

        for fname in ref_wav_list:
            audio_ref, sr = librosa.load(fname, sr=self.hps.data.sampling_rate)
            y = torch.FloatTensor(audio_ref)
            y = y.to(self.device)
            y = y.unsqueeze(0)
            y = self._prepare_spectrogram(y)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())
        gs = torch.stack(gs).mean(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs.cpu(), se_save_path)

        return gs

    def convert(
        self,
        audio_src_path,
        src_se,
        tgt_se,
        output_path=None,
        tau=0.3,
        message="default",
    ):
        audio, sample_rate = librosa.load(
            audio_src_path, sr=self.hps.data.sampling_rate
        )
        audio = torch.tensor(audio).float()

        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            y = self._prepare_spectrogram(y)
            spec_lengths = torch.LongTensor([y.size(-1)]).to(self.device)
            audio = (
                self.model.voice_conversion(
                    y, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )

            if output_path is None:
                return audio
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
