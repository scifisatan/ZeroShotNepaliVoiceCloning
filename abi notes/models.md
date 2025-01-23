# Text To Speeech Model on models.py

## TextEncoder:

- Encodes textual inputs into latent representations.
- Uses an embedding layer and attention-based encoder, projecting the output into a feature space for subsequent components.

## Duration Predictors:

- Two predictors (DurationPredictor and StochasticDurationPredictor) estimate the duration of phonemes or speech units.
- StochasticDurationPredictor uses normalizing flows for flexible modeling of duration distributions.

## PosteriorEncoder:

- Encodes mel-spectrogram features into latent representations.
- Outputs a distribution over the latent space to sample features for synthesis.

## Generator:

- Converts latent features into audio waveforms using transposed convolutions (upsampling) and residual blocks.
- The generator is the heart of waveform generation in this TTS model.

## ReferenceEncoder:

- Processes reference audio features into latent embeddings.
- Includes convolutional layers and GRU to extract temporal and frequency-based patterns.

## Residual Coupling Block:

- Implements normalizing flow layers for flexible latent transformations, ensuring invertibility.
- Alternates between coupling and flipping operations.

## SynthesizerTrn:

- Combines all components into a comprehensive model for training.
- Handles text encoding, speaker conditioning (via speaker embeddings), flow-based latent processing, and waveform synthesis.
