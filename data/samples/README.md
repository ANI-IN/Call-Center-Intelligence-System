# Sample Call Center Audio Dataset

10 sample call center recordings for testing and evaluation.

## Files

| File | Description |
|------|-------------|
| `sample_01.mp3` | Short customer service call |
| `sample_02.mp3` | Extended billing inquiry |
| `sample_03.mp3` | Technical support call |
| `sample_04.mp3` | Account management call |
| `sample_05.mp3` | Service complaint |
| `sample_06.mp3` | Product inquiry |
| `sample_07.mp3` | Subscription issue |
| `sample_08.mp3` | Payment dispute |
| `sample_09.mp3` | Service cancellation |
| `sample_10.mp3` | General inquiry |

## Usage

Upload any of these files to the app's "Analyze Call" tab, or use them for batch processing.

```bash
# Run the app locally
python app.py

# Or process via the pipeline directly
python -c "
from src.services.pipeline import process_call
# ... see README.md for full setup
"
```

## Format

- **Format:** MP3
- **Channels:** Mono/Stereo
- **Sample Rate:** Various (Whisper handles resampling)
- **Total Size:** ~30 MB

## License

These samples are provided for educational and testing purposes only.
