import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

def spectral_normalize_torch(magnitudes, C=1, clip_val=1e-5):
    """
    Apply dynamic range compression to the input tensor.
    Then applies spectral normalization to the input magnitudes.
    
    Parameters:
    x (Tensor): Input tensor.
    C (float): Compression factor. Default is 1.
    clip_val (float): Minimum value to clamp the input tensor. Default is 1e-5.
    
    Returns:
    Tensor: Normalized magnitudes.
    """

    output = torch.log(torch.clamp(magnitudes, min=clip_val) * C)
    return output


def spectral_de_normalize_torch(magnitudes, C=1):
    """
    Apply dynamic range decompression to the input tensor.
    Then applies spectral denormalization to the input magnitudes.
    
    Parameters:
    magnitudes (Tensor): Input tensor.
    C (float): Compression factor used during compression. Default is 1.
    
    Returns:
    Tensor: Denormalized magnitudes.
    """
    output = torch.exp(magnitudes) / C
    return output

mel_basis = {}  # Dictionary to cache computed Mel filter banks
hann_window = {}  # Dictionary to cache computed Hann windows


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
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