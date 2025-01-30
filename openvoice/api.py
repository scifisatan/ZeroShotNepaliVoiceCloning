import torch
import torch.utils.data
import soundfile
from openvoice import utils
from openvoice.models import VoiceConverter
import os
import librosa

class ToneColorConverter(object):
    def __init__(self, config_path, device='cuda:0'):
        if 'cuda' in device:
            assert torch.cuda.is_available()
            
        self.hps = utils.get_hparams_from_file(config_path)
        self.device = device
        
        # Initialize VoiceConvert model specifically for tone color conversion
        self.model = VoiceConverter(
            len(getattr(self.hps, 'symbols', [])),
            self.hps.data.filter_length // 2 + 1,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        ).to(device)
        
        self.model.eval()
        self.version = getattr(self.hps, '_version_', "v1")

    def load_ckpt(self, ckpt_path):
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print("Loaded checkpoint '{}'".format(ckpt_path))
        print('missing/unexpected keys:', a, b)

    def extract_se(self, ref_wav_list, se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]
        
        gs = []
        
        for fname in ref_wav_list:
            audio_ref, sr = librosa.load(fname, sr=self.hps.data.sampling_rate)
            y = torch.FloatTensor(audio_ref)
            y = y.to(self.device)
            y = y.unsqueeze(0)
            y = self.spectrogram_torch(y, self.hps.data.filter_length,
                                self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                center=False).to(self.device)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())
        gs = torch.stack(gs).mean(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs.cpu(), se_save_path)

        return gs

  

    def spectrogram_torch(self, y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # Warn if signal exceeds typical [-1,1] range
        if torch.min(y) < -1.1:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.1:
            print("max value is ", torch.max(y))

        global hann_window
        # Create a unique key identifying (win_size, dtype, device)
        dtype_device = str(y.dtype) + "_" + str(y.device)
        wnsize_dtype_device = str(win_size) + "_" + dtype_device

        # If not already computed, create and store Hann window for current config
        if wnsize_dtype_device not in hann_window:
            hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
                dtype=y.dtype, device=y.device
            )

        # Pad the signal so frames are aligned for STFT
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)

        # Compute STFT (real + imaginary parts) with Hann window
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        # Convert to magnitude by summing squares of real and imaginary parts
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec

    def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3, message="default"):
        # load audio
        audio, sample_rate = librosa.load(audio_src_path, sr=self.hps.data.sampling_rate)
        audio = torch.tensor(audio).float()
        
        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            spec = self.spectrogram_torch(y, self.hps.data.filter_length,
                                    self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                                    center=False).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio = self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][
                        0, 0].data.cpu().float().numpy()
            
            if output_path is None:
                return audio
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)