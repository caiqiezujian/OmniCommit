# Edit inside Flex_web, then run: bash Fourth/start_http.sh
CUDA_VISIBLE_DEVICES=6
ASR_MODEL='/data/yb/Code/OmniCommit/output/fourth-merged-1s-v2/v0-20261121-063241/sft-checkpoint-2721'

# Edit only the last committed period in the caller's whole packet.
# Chinese/English: below 150 ms remove, 150 to below 450 ms use a comma.
# Other or ambiguous text keeps the existing comma rule up to 300 ms.
PACKET_PERIOD_TO_COMMA=false
PACKET_PERIOD_THRESHOLD_MS=450
PACKET_PERIOD_REMOVE_THRESHOLD_MS=150

# readable writes one events.jsonl per meeting session.
# detailed also records model inputs, raw replies and diagnostics in trace.jsonl.
HTTP_EVENT_LOG_MODE=readable

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
