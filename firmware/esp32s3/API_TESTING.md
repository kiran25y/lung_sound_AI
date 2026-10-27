# ESP32-S3 audio inference API

This local Flask service accepts a WAV upload and sends normalized PCM16 to
the ESP32-S3. The deployed firmware—not Flask—runs the log-Mel preprocessing,
quality gate, and TFLite Micro model.

## Setup

From PowerShell in this repository:

```powershell
python -m pip install -r tools\requirements-api.txt
```

## Start the API

Close any serial monitor first because only one program can own `COM4`.

```powershell
python -u tools\esp32_inference_api.py --serial-port COM4
```

The server listens only on `http://127.0.0.1:5000` by default.

## Test a WAV through the ESP32

```powershell
curl.exe -X POST http://127.0.0.1:5000/infer `
  -F "audio=@C:\path\to\lung_sound.wav"
```

The API accepts uncompressed PCM16 WAV files. Stereo is mixed to mono, other
sample rates are resampled to 16 kHz, short recordings are repeated, and the
first four seconds are sent to the board.

A successful response contains `decision: pass`, two-class and three-class
labels, and crackle/wheeze probabilities. A quality rejection contains
`decision: retry` and the gate metrics instead of a prediction.

## Test the attached I2S microphone

```powershell
curl.exe -X POST http://127.0.0.1:5000/infer/live
```

The microphone is configured for `WS=GPIO1`, `SCK/BCLK=GPIO2`, and `SD=GPIO4`.
The board records six seconds at 16 kHz and classifies the middle four seconds.
The corrected mapping has been verified with non-zero PCM. If the captured
signal is too quiet or poorly coupled, this route returns
`low_signal_quality`; place the sensor securely on a clear sound source and
retry. A model prediction is returned only after the quality gate passes.

This API is for engineering validation and is not a medical diagnostic service.
