from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from pickle import load
from typing import Any
import numpy as np
from numpy.typing import NDArray
import onnxruntime # type: ignore
from piper_phonemize import phonemize_espeak, phonemize_codepoints
from ..utils.resources import resource_path
from .phonemizer import Phonemizer


class Synthesizer:
    """Synthesizer, based on the VITS model.

    Trained using the Piper project (https://github.com/rhasspy/piper)

    Attributes:
    -----------
    session: onnxruntime.InferenceSession
        The loaded VITS model.
    id_map: dict
        A dictionary mapping phonemes to ids.

    Methods:
    --------
    __init__(self, model_path, use_cuda):
        Initializes the Synthesizer class, loading the VITS model.

    generate_speech_audio(self, text):
        Generates speech audio from the given text.

    _phonemizer(self, input_text):
        Converts text to phonemes using espeak-ng.

    _phonemes_to_ids(self, phonemes):
        Converts the given phonemes to ids.

    _synthesize_ids_to_raw(self, phoneme_ids, speaker_id, length_scale, noise_scale, noise_w):
        Synthesizes raw audio from phoneme ids.

    say_phonemes(self, phonemes):
        Converts the given phonemes to audio.
    """

    # Constants
    MAX_WAV_VALUE = 32767.0

    # Settings
    MODEL_PATH = resource_path("models/TTS/vietnamese.onnx")
    CONFIG_PATH = resource_path("models/TTS/vietnamese.json")

    # Conversions
    PAD = "_"  # padding (0)
    BOS = "^"  # beginning of sentence
    EOS = "$"  # end of sentence

    def __init__(
        self, model_path: Path = MODEL_PATH, config_path: Path = CONFIG_PATH
    ) -> None:
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.sample_rate = self.config["audio"]["sample_rate"]

    def audio_float_to_int16(self,
            audio: np.ndarray, max_wav_value: float = 32767.0
    ) -> np.ndarray:
        """Normalize audio and convert to int16 range"""
        audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
        audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
        audio_norm = audio_norm.astype("int16")
        return audio_norm

    def phonemize(self, text):
        if self.config["phoneme_type"] == "espeak":
            return phonemize_espeak(text, self.config["espeak"]["voice"])
        elif self.config["phoneme_type"] == "text":
            return phonemize_codepoints(text)
        else:
            raise ValueError(f"Unsupported phoneme type: {self.config['phoneme_type']}")

    def phonemes_to_ids(self, phonemes):
        id_map = self.config["phoneme_id_map"]
        ids = [id_map["^"][0]]  # Start of sentence (BOS)
        for phoneme in phonemes:
            ids.extend(id_map.get(phoneme, []))
            ids.extend(id_map["_"])  # PAD between phonemes
        ids.append(id_map["$"][0])  # End of sentence (EOS)
        return ids

    def synthesize_audio(self, text):

        phoneme_groups = self.phonemize(text)

        audio_segments = []
        for phonemes in phoneme_groups:
            phoneme_ids = self.phonemes_to_ids(phonemes)
            text_tensor = np.array([phoneme_ids], dtype=np.int64)
            text_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
            scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)  # default scales

            inputs = {
                "input": text_tensor,
                "input_lengths": text_lengths,
                "scales": scales,
            }

            if self.config["num_speakers"] > 1:
                inputs["sid"] = np.array([0], dtype=np.int64)  # default speaker ID

            audio = self.session.run(None, inputs)[0].squeeze()
            print("AUDIO: ", audio)
            audio_int16 = self.audio_float_to_int16(audio)
            audio_segments.append(audio_int16)

        full_audio = np.concatenate(audio_segments)
        return  full_audio

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "ort_sess"):
            del self.ort_sess

