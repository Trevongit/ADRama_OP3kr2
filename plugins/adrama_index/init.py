"""
ADRama IndexTTS2 Plugin: Streamlined zero-shot TTS with emo control.
Import: from adrama_index import IndexTTSEngine
"""

from .core import IndexTTS2Core
from .engine import IndexTTSEngine

__version__ = "0.1.0-adrama"
__all__ = ["IndexTTS2Core", "IndexTTSEngine"]
