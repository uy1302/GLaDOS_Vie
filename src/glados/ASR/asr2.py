from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore
import soundfile as sf  # type: ignore
from transformers import Wav2Vec2Processor  # type: ignore

from ..utils.resources import resource_path

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)


class AudioTranscriber:
    MODEL_PATH = resource_path("models/ASR/viet_asr_model.onnx")
    PROCESSOR_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    SAMPLE_RATE = 16000

    def __init__(
            self,
            model_path: Path = MODEL_PATH,
            processor_name: str = PROCESSOR_NAME
    ) -> None:
        """
        Initialize a VietnameseASRTranscriber with ONNX model.

        Parameters:
            model_path (Path, optional): Path to the ONNX model file.
                Defaults to the predefined MODEL_PATH.
            processor_name (str, optional): Name of the Wav2Vec2 processor to use.
                Defaults to the predefined PROCESSOR_NAME.
            use_cuda (bool, optional): Whether to use CUDA for inference if available.
                Defaults to True.

        Initializes the transcriber by:
            - Configuring ONNX Runtime providers based on CUDA availability
            - Creating inference session with the specified model
            - Loading the Wav2Vec2Processor for audio feature extraction and tokenization
        """

        # Session options
        sess_options = ort.SessionOptions()

        # Create ONNX session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options
        )

        # Load the processor from Hugging Face
        self.processor = Wav2Vec2Processor.from_pretrained(processor_name)


    def process_audio(self, audio: NDArray[np.float32]) -> dict[str, np.ndarray]:
        """
        Process audio input to prepare it for model inference.

        Parameters:
            audio (NDArray[np.float32]): Input audio time series data as a numpy float32 array

        Returns:
            dict[str, np.ndarray]: Dictionary containing processed model inputs
        """
        # Extract features using the processor
        inputs = self.processor(
            audio,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="np"
        )

        return {
            "input_values": inputs.input_values
        }

    def transcribe(self, audio: NDArray[np.float32]) -> Union[str, tuple[str, str]]:
        """
        Transcribes an audio signal to text using the Vietnamese ASR model.

        Parameters:
            audio (NDArray[np.float32]): Input audio signal as a numpy float32 array.
            compare_with_pytorch (bool, optional): Whether to compare results with PyTorch model.
                Defaults to False.

        Returns:
            Union[str, tuple[str, str]]: Transcribed text, or a tuple of (onnx_result, pytorch_result) if comparing.
        """
        # Process audio
        inputs = self.process_audio(audio)

        # Run ONNX inference
        ort_inputs = {self.session.get_inputs()[0].name: inputs["input_values"]}
        ort_outputs = self.session.run(None, ort_inputs)

        # Convert logits to text
        predicted_ids = np.argmax(ort_outputs[0], axis=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription

    def transcribe_file(self, audio_path: str) -> Union[str, tuple[str, str]]:
        """
        Transcribe an audio file to text.

        Parameters:
            audio_path (str): Path to the audio file to be transcribed.
            compare_with_pytorch (bool, optional): Whether to compare with PyTorch model.
                Defaults to False.

        Returns:
            Union[str, tuple[str, str]]: Transcribed text, or tuple of results if comparing.

        Raises:
            FileNotFoundError: If the specified audio file does not exist.
            ValueError: If the audio file cannot be read or processed.
        """
        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")

        # Resample to 16kHz if needed
        if sr != self.SAMPLE_RATE:
            import librosa  # type: ignore
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)

        return self.transcribe(audio)

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "session"):
            del self.session