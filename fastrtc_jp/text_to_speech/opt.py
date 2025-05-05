
from dataclasses import dataclass
from fastrtc.text_to_speech.tts import TTSOptions as FastRTC_TTSOptions

@dataclass
class SpkOptions(FastRTC_TTSOptions):
    model:int|str|None = None
    speaker_id: int|None = None
    speaker_style: str|None = None
    speaker_name: str|None = None
    split:bool = False
    speedScale: float = 1.0
    pitchOffset: float = 0.0
    lang: str = "ja-jp"

    def lang2str(self) -> str:
        if self.lang.startswith("ja"):
            return "ja"
        if self.lang.startswith("en"):
            return "en"
        if len(self.lang) >= 2:
            return self.lang[:2]
        return self.lang