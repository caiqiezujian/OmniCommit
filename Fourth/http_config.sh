# Edit inside Flex_web, then run: bash Fourth/start_http.sh
CUDA_VISIBLE_DEVICES=6
ASR_MODEL='/data/yb/Code/OmniCommit/output/fourth-merged-1s-v2/v0-20261121-063241/sft-checkpoint-2721'

# Set true to replace a packet-final sentence period with a comma when the
# caller reports less than 300 ms of silence for the packet. Audio alone is unchanged.
PACKET_PERIOD_TO_COMMA=false

# Relative paths in this configuration start in Fourth/.
HTTP_ARGS=(
  --host 0.0.0.0
  --port 18892
  --backend vllm
  --protocol speaker
  --chunk-ms 1000
  --window-seconds 60
  --discard-seconds 50
  --max-context-tokens 4352
  --max-new-tokens 512
  --prefix-caching
  --no-recovery
  --max-meetings 16
  --max-in-flight 8
  --batch-size 4
  --max-packet-seconds 8
  --request-log logs/http-requests.jsonl
  --debug-dir logs/http-debug
)
