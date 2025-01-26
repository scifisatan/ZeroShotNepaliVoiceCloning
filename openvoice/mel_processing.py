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

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    # Convert a linear-frequency spectrogram to a Mel spectrogram
    global mel_basis
    # Unique key for caching Mel filter bank
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device

    # If not in cache, compute and store the Mel filter bank
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    # Transform spectrogram to Mel scale
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # Normalize the resulting Mel spectrogram
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    # Warn if signal exceeds typical [-1.0,1.0] range
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    # Unique keys for Mel filter bank and Hann window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device

    # Create Mel filter bank if missing
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    # Create Hann window if missing
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # Pad and reshape input
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # Compute STFT
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
    # Convert STFT to magnitude
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    # Map to Mel scale
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    # Normalize the Mel spectrogram
    spec = spectral_normalize_torch(spec)

    return spec