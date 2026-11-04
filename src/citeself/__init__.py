"""CiteSelf — an AI of you that only speaks on the record."""
__version__ = "0.1.0.dev0"

from .engine import Answer, Conversation, Persona  # noqa: E402
from .validate import Segment, validate_segments  # noqa: E402

__all__ = ["Answer", "Conversation", "Persona", "Segment", "validate_segments", "__version__"]
