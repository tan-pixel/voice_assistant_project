Place the custom openWakeWord model for the `Hey Atlas` wake phrase in this folder.

Supported default filenames:
- `hey_atlas.onnx`
- `hey_atlas.tflite`
- `hey-atlas.onnx`
- `hey-atlas.tflite`

You can also point the app at a different custom model path by setting:

```bash
M2_OPENWAKEWORD_MODEL_PATH=/absolute/or/relative/path/to/hey_atlas.onnx
```

Optional runtime tuning:

```bash
M2_OPENWAKEWORD_VAD_THRESHOLD=0.5
```

The pipeline will not fall back to a built-in wake phrase such as `hey jarvis`.
