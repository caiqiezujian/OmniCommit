# Edit this file inside Flex_web, then launch with bash Fourth/start_server.sh.
CUDA_VISIBLE_DEVICES=6
ASR_MODEL='/data/yb/Code/OmniCommit/output/fourth-merged-1s-v2/v0-20261121-063241/sft-checkpoint-2721'

# Set this to the existing ONNX file visible inside the container.
# Set to '' to switch off pre-decode silence filtering.
SILERO_VAD_MODEL='/data/yb/models/silero_vad.onnx'
# Frames scoring above this value keep the current ASR turn eligible for decoding.
SILERO_QUIET_THRESHOLD=0.10

WEB_ARGS=(
  --backend vllm
  --protocol speaker
  --chunk-ms 1000
  --window-seconds 60
  --discard-seconds 50
  --max-context-tokens 4352
  --max-new-tokens 512
  --recovery
  --host 0.0.0.0
  --port 18895
  --https-host 10.185.1.71
  --cert-dir /data06/yb/Code/OmniCommit/certs
)
