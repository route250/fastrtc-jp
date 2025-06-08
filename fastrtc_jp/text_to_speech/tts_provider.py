
from typing import Type, AsyncGenerator
from functools import lru_cache

from logging import getLogger

from fastrtc.text_to_speech.tts import TTSModel

from fastrtc_jp.text_to_speech.gtts import GTTSModel, GTTSOptions
from fastrtc_jp.text_to_speech.opt import SpkOptions

logger = getLogger(__name__)

class TtsProvider:

    @staticmethod
    def get_tts_model(options:SpkOptions) -> TTSModel:
        """Return a TTSModel instance.

        This simple example ignores ``class_id`` and always returns ``GTTSModel``.
        ``options`` is currently unused but kept for signature compatibility.
        """
        print(f"get_tts_model: class_id={options.class_id}", flush=True)
        if options.class_id == "voicevox":
            from fastrtc_jp.text_to_speech.voicevox import VoicevoxTTSModel
            return VoicevoxTTSModel()
        elif options.class_id == "sbv2":
            from fastrtc_jp.text_to_speech.style_bert_vits2 import StyleBertVits2,SBV2_MODELS
            return StyleBertVits2()
        else:
            logger.warning(f"get_tts_model: Unknown class_id {options.class_id}, using GTTSModel.")
        return GTTSModel()

    @staticmethod
    def get_tts_options(options:SpkOptions) -> SpkOptions:
        """Return ``SpkOptions`` for the given class.

        The sample implementation converts the generic :class:`SpkOptions` into
        :class:`GTTSOptions` used by :class:`GTTSModel`.
        """
        if options.class_id == "voicevox":
            from fastrtc_jp.text_to_speech.voicevox import VoicevoxTTSOptions
            opts = VoicevoxTTSOptions()
            opts.speaker_id = options.speaker_id or 8  # Default to speaker ID 8 if not specified
            return opts
        elif options.class_id == "sbv2":
            from fastrtc_jp.text_to_speech.style_bert_vits2 import StyleBertVits2Options
            opts = StyleBertVits2Options()
            opts.model = options.model
            opts.speaker_id = options.speaker_id
        else:
            opts = GTTSOptions()
        opts.lang = options.lang
        opts.speedScale = options.speedScale
        opts.pitchOffset = options.pitchOffset
        return opts

