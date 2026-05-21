# Disco

- mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit
- ibm-granite/granite-speech-4.1-2b
- mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32
- mlx-community/translategemma-4b-it-8bit
- mlx-community/Qwen3.5-0.8B-MLX-8bit

```sh
 uv run disco-web -d 7 -k --language english
 uv run disco-web -d 7 --asr-backend granite-speech --language english
```
