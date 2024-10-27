MODEL_PATH="outputs/magictailor"
PROMPT="<v0> with <v1>, on the beach"
OUTPUT_PATH="outputs/inference/result.jpg"

python inference.py \
  --model_path $MODEL_PATH \
  --prompt $PROMPT \
  --output_path $OUTPUT_PATH
  