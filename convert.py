import os
import sys
import torch
import azure.cognitiveservices.speech as speechsdk
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from dotenv import load_dotenv


def load_env_variables():
    load_dotenv()
    subscription_key = os.getenv("AZURE_SPEECH_KEY")
    service_region = os.getenv("AZURE_SERVICE_REGION")

    if not subscription_key or not service_region:
        sys.exit(
            "Error: Missing Azure Speech configuration. Please set AZURE_SPEECH_KEY and AZURE_SERVICE_REGION environment variables."
        )

    return subscription_key, service_region


def initialize_tone_converter(device="cuda:0" if torch.cuda.is_available() else "cpu"):
    ckpt_converter = "checkpoints_v2/converter"
    converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
    converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
    return converter


def text_to_speech(text, speech_output_path, subscription_key, service_region):
    speech_config = speechsdk.SpeechConfig(
        subscription=subscription_key, region=service_region
    )
    speech_config.speech_synthesis_voice_name = "ne-NP-HemkalaNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(filename=speech_output_path)
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized to [{speech_output_path}] for text [{text}]")
    else:
        sys.exit("Speech synthesis failed!")


def extract_embedding(reference_path, converter):
    return se_extractor.get_se(
        reference_path, converter, target_dir="processed", vad=True
    )


def perform_tone_conversion(
    base_tts_output_path, source_se, target_se, output_path, converter
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    converter.convert(
        audio_src_path=base_tts_output_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_path,
    )
    print(f"Tone color conversion complete! Output saved to {output_path}")


def main():
    subscription_key, service_region = load_env_variables()
    tone_color_converter = initialize_tone_converter()

    while True:
        reference_path = input(
            "Enter the path to the reference audio (or type 'exit' to quit): "
        ).strip()
        if reference_path.lower() == "exit":
            print("Exiting...")
            break

        target_se, _ = extract_embedding(reference_path, tone_color_converter)
        source_se = None
        base_tts_output_path = "processed/base_tts_output.wav"

        while True:
            text_input = input(
                "Enter text for speech (or 'back' to change reference, 'exit' to quit): "
            ).strip()
            if text_input.lower() == "exit":
                sys.exit("Exiting...")
            elif text_input.lower() == "back":
                break

            os.makedirs(os.path.dirname(base_tts_output_path), exist_ok=True)
            text_to_speech(
                text_input, base_tts_output_path, subscription_key, service_region
            )

            if source_se is None:
                source_se, _ = extract_embedding(
                    base_tts_output_path, tone_color_converter
                )

            output_path = "output/converted.wav"
            perform_tone_conversion(
                base_tts_output_path,
                source_se,
                target_se,
                output_path,
                tone_color_converter,
            )


if __name__ == "__main__":
    main()
