"""Local HF and vLLM inference, streaming context and media handling for Fourth."""

DEFAULT_MAX_NEW_TOKENS = 512
# Keep room for the same 3840-token multimodal prompt after reserving output.
DEFAULT_MAX_CONTEXT_TOKENS = 3840 + DEFAULT_MAX_NEW_TOKENS
