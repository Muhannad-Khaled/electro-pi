import logging

from livekit.agents import stt
from livekit.agents.stt import (
    STTCapabilities,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger("stt_stub")


class SttStub(stt.STT):
    """Stub STT implementing the official livekit.agents.stt.STT base class.

    Bypasses actual audio processing — returns a pre-set text transcript so
    the AgentSession pipeline shape (session → recognize() → _recognize_impl())
    is preserved without needing a microphone or a live STT provider.
    """

    def __init__(self, transcript: str = "") -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=False, interim_results=False)
        )
        self._transcript = transcript

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        lang = language if language is not NOT_GIVEN else ""
        logger.info('[STT] Transcribed user audio → "%s"', self._transcript)
        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(language=lang, text=self._transcript)],
        )
