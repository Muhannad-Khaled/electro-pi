import logging
import struct

from livekit.agents import tts
from livekit.agents.tts import (
    AudioEmitter,
    ChunkedStream,
    TTSCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("tts_stub")

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
_SILENCE_DURATION_S = 0.1  # 100 ms of silent audio so the pipeline doesn't stall


class _StubChunkedStream(ChunkedStream):
    """Logs the text that would be spoken, emits a brief silent PCM frame."""

    async def _run(self, output_emitter: AudioEmitter) -> None:
        logger.info('[TTS] Synthesizing speech ← "%s"', self._input_text)
        output_emitter.initialize(
            request_id="stub-tts",
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        num_samples = int(SAMPLE_RATE * _SILENCE_DURATION_S)
        silent_bytes = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        output_emitter.push(silent_bytes)


class TtsStub(tts.TTS):
    """Stub TTS implementing the official livekit.agents.tts.TTS base class.

    Logs output text and emits silent audio so the pipeline completes its TTS
    stage cleanly without any audio hardware or provider credentials.
    """

    def __init__(self) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _StubChunkedStream:
        return _StubChunkedStream(tts=self, input_text=text, conn_options=conn_options)
