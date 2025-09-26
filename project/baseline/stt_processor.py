import logging

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SenseVoiceSTT:
    def __init__(self, model_id: str = "FunAudioLLM/SenseVoiceSmall", device: str = "cuda"):
        """
        SenseVoice STT 모델을 초기화하고 로드합니다.

        Args:
            model_id: ModelScope에서 사용할 모델 ID
            device: 사용할 디바이스 ("cuda" 또는 "cpu")
        """
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()

    def _load_model(self):
        """STT 모델을 로드합니다."""
        logger.info(f"Loading SenseVoice STT model: {self.model_id} on device: {self.device}")
        try:
            self.model = AutoModel(
                model=self.model_id,
                trust_remote_code=True,
                device=self.device,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                hub="hf",
            )
            logger.info("SenseVoice STT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SenseVoice STT model: {e}")
            raise

    def transcribe(self, audio_path: str) -> str:
        """
        주어진 경로의 오디오 파일을 텍스트로 변환합니다.

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            변환된 텍스트
        """
        if not self.model:
            logger.error("STT model is not loaded.")
            return "Error: STT model not loaded."

        logger.info(f"Transcribing audio file: {audio_path}")
        try:
            result = self.model.generate(
                input=audio_path,
                cache={},
                language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False,
                hub="hf",
            )
            transcribed_text = rich_transcription_postprocess(result[0]["text"])
            logger.info(f"Transcription successful. Result: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return f"Error: Transcription failed. {e}"


# 단독 테스트용
if __name__ == "__main__":
    # 이 파일을 직접 실행할 경우, 테스트 오디오 파일로 STT 기능을 시험합니다.
    # 예: python stt_processor.py

    # 테스트용 가상 오디오 파일 생성 (실제로는 실제 .wav 파일 경로를 사용해야 함)
    import os

    import numpy as np
    import soundfile as sf

    test_audio_path = "/home/elicer/resources/output.wav"
    testfile_generated = False
    if not os.path.exists(test_audio_path):
        logger.info(f"Creating a dummy audio file for testing: {test_audio_path}")
        testfile_generated = True
        samplerate = 16000
        duration = 2  # seconds
        frequency = 440  # Hz
        t = np.linspace(0.0, duration, int(samplerate * duration), endpoint=False)
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2.0 * np.pi * frequency * t)
        sf.write(test_audio_path, data.astype(np.int16), samplerate)

    stt = SenseVoiceSTT()
    text = stt.transcribe(test_audio_path)
    print(f"Transcribed text: {text}")

    # 테스트 후 생성된 파일 삭제
    if os.path.exists(test_audio_path) and testfile_generated:
        os.remove(test_audio_path)
