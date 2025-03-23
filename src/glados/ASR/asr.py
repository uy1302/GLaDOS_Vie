from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore
import soundfile as sf  # type: ignore
from transformers import WhisperProcessor  # type: ignore

from ..utils.resources import resource_path

# Default OnnxRuntime is way to verbose
ort.set_default_logger_severity(4)


class AudioTranscriber:
    ENCODER_MODEL_PATH = resource_path("models/ASR/encoder_model_quantized.onnx")
    DECODER_MODEL_PATH = resource_path("models/ASR/decoder_model_quantized.onnx")
    DECODER_WITH_PAST_MODEL_PATH = resource_path("models/ASR/decoder_with_past_model_quantized.onnx")
    PROCESSOR_NAME = "openai/whisper-medium"
    SAMPLE_RATE = 16000

    def __init__(
            self,
            encoder_model_path: Path = ENCODER_MODEL_PATH,
            decoder_model_path: Path = DECODER_MODEL_PATH,
            decoder_with_past_model_path: Path = DECODER_WITH_PAST_MODEL_PATH,
            processor_name: str = PROCESSOR_NAME,
    ) -> None:
        """
        Initialize a PhoWhisperTranscriber with ONNX encoder and decoder models.

        Parameters:
            encoder_model_path (Path, optional): Path to the ONNX encoder model file.
                Defaults to the predefined ENCODER_MODEL_PATH.
            decoder_model_path (Path, optional): Path to the ONNX decoder model file.
                Defaults to the predefined DECODER_MODEL_PATH.
            decoder_with_past_model_path (Path, optional): Path to the ONNX decoder with past key-values model file.
                Defaults to the predefined DECODER_WITH_PAST_MODEL_PATH.
            processor_name (str, optional): Name of the Whisper processor to use.
                Defaults to the predefined PROCESSOR_NAME.

        Initializes the transcriber by:
            - Configuring ONNX Runtime providers, excluding TensorRT if available
            - Creating inference sessions with the specified encoder and decoder models
            - Loading the WhisperProcessor for audio feature extraction and tokenization
            - Configuring runtime parameters for inference

        Note:
            - Removes TensorRT and CoreML execution providers to ensure compatibility across hardware
            - Uses default model paths and processor name if not explicitly specified
        """
        # providers = ort.get_available_providers()
        # if "TensorrtExecutionProvider" in providers:
        #     providers.remove("TensorrtExecutionProvider")
        # if "CoreMLExecutionProvider" in providers:
        #     providers.remove("CoreMLExecutionProvider")

        # Create ONNX sessions for encoder and decoder
        self.encoder_session = ort.InferenceSession(
            encoder_model_path
        )

        self.decoder_session = ort.InferenceSession(
            decoder_model_path
        )

        # Load the processor from Hugging Face
        self.processor = WhisperProcessor.from_pretrained(processor_name)

        # Maximum decoding length
        self.max_length = 448

    def process_audio(self, audio: NDArray[np.float32]) -> dict[str, np.ndarray]:
        """
        Process audio input to prepare it for model inference.

        This method transforms raw audio data into model inputs by:
        - Extracting features using the WhisperProcessor
        - Creating initial decoder input token IDs
        - Packaging the inputs into a dictionary suitable for model inference

        Parameters:
            audio (NDArray[np.float32]): Input audio time series data as a numpy float32 array

        Returns:
            dict[str, np.ndarray]: Dictionary containing processed model inputs:
                - "input_features": Audio features with shape [1, n_mels, time]
                - "decoder_input_ids": Initial decoder input token IDs

        Notes:
            - Uses the WhisperProcessor for feature extraction
            - Sets up the decoder with start token (token_id=1)
        """
        # Extract features using the processor
        input_features = self.processor(
            audio,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="np"
        ).input_features

        # Prepare decoder inputs (start token)
        decoder_input_ids = np.array([[1]], dtype=np.int64)

        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids
        }


    def transcribe(self, audio: NDArray[np.float32]) -> str:
        """
        Transcribes an audio signal to text using the Pho Whisper models.

        Processes the input audio into features, runs inference through both encoder and decoder
        ONNX Runtime sessions, and decodes the output token IDs into a human-readable transcription.

        Parameters:
            audio (NDArray[np.float32]): Input audio signal as a numpy float32 array.

        Returns:
            str: Transcribed text representation of the input audio.

        Notes:
            - Requires pre-initialized ONNX Runtime sessions for encoder and decoder
            - Uses greedy decoding for token generation
            - Supports Vietnamese speech recognition through Pho Whisper models
        """
        # Process audio
        inputs = self.process_audio(audio)

        # Run encoder
        encoder_outputs = self.encoder_session.run(
            None,
            {"input_features": inputs["input_features"]}
        )[0]

        # Initialize for decoding
        decoder_input_ids = inputs["decoder_input_ids"]
        generated_ids = [decoder_input_ids[0, 0]]

        # Greedy decoding
        for _ in range(self.max_length):
            # Create decoder inputs
            decoder_inputs = {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_outputs
            }

            # Run decoder
            logits = self.decoder_session.run(None, decoder_inputs)[0]

            # Get the next token
            next_token_id = np.argmax(logits[:, -1, :], axis=-1)
            next_token_id = np.array([[next_token_id[0]]], dtype=np.int64)

            # Update decoder inputs
            decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id], axis=1)

            # Add to generated ids
            generated_ids.append(next_token_id[0, 0])

            # Check for end of sequence
            if next_token_id[0, 0] == 50257:  # EOS token
                break

        # Decode the generated tokens
        transcription = self.processor.decode(generated_ids, skip_special_tokens=True)
        return transcription

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text by reading the audio data and processing it through the model.

        Parameters:
            audio_path (str): Path to the audio file to be transcribed.

        Returns:
            str: The transcribed text content of the audio file.

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
        """Clean up ONNX sessions to prevent context leaks."""
        if hasattr(self, "encoder_session"):
            del self.encoder_session
        if hasattr(self, "decoder_session"):
            del self.decoder_session
