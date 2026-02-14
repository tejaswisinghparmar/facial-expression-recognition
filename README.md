# AI Vision: Emotion & ASL Recognition

A powerful desktop application that combines real-time emotion detection and American Sign Language (ASL) recognition using advanced computer vision and deep learning models.

## Features

- **Emotion Detection**: Real-time facial emotion recognition (Happy, Sad, Fear, Disgust, Angry, Surprise)
- **ASL Recognition**: Hand gesture recognition for American Sign Language finger-spelling
- **Modern GUI**: Dark-themed interface built with CustomTkinter for a sleek, professional look
- **AI Integration**: Powered by Ollama for advanced language model capabilities
- **Real-time Processing**: Live video stream analysis with minimal latency

## Tech Stack

- **Computer Vision**: OpenCV, MediaPipe, DeepFace
- **GUI Framework**: CustomTkinter
- **Deep Learning**: TensorFlow/Keras, DeepFace
- **Hand Recognition**: MediaPipe Hand Landmarker (Tasks API)
- **LLM Integration**: Ollama
- **Image Processing**: Pillow

## Requirements

- Python 3.8 to 3.10(compulsory)
- Webcam for real-time video input
- ~500MB+ disk space for ML models
- Ollama installed and running (for LLM features)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/tejaswisinghparmar/facial-expression-recognition.git
cd facial-expression-recognition
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Hand Landmarker Model
The application requires the MediaPipe Hand Landmarker model file (`hand_landmarker.task`). Ensure this file is in the project root directory:
```
my_ai_project/
├── hand_landmarker.task  # Required model file
├── app.py
├── main.py
└── modules/
```

[Download hand_landmarker.task](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

### 5. Install & Start Ollama
For LLM integration features, install [Ollama](https://ollama.ai) and start the service before running the application.

## Usage

### Running the Application
```bash
python main.py
```

This launches the GUI with two primary modes:

#### Emotion Detection Mode
- Real-time detection of facial emotions
- Color-coded emotion boxes with confidence scores
- Smooth frame updates at camera refresh rate

#### ASL Recognition Mode
- Hand gesture tracking and finger-spelling recognition
- Hand skeleton visualization for feedback
- Letter-by-letter ASL input capture

## Project Structure

```
my_ai_project/
├── app.py                   # Main GUI application (CustomTkinter)
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── hand_landmarker.task     # MediaPipe hand recognition model
│
└── modules/
    ├── __init__.py
    ├── emotion_detector.py  # DeepFace-based emotion detection
    ├── asl_recognizer.py    # Hand landmarks & ASL classification
    └── ollama_client.py     # Ollama LLM integration
```

## Module Details

### `emotion_detector.py`
- Detects 7 emotion classes (including neutral)
- Uses DeepFace for face detection and analysis
- Returns annotated frames with emotion labels and confidence scores

### `asl_recognizer.py`
- MediaPipe Hand Landmarker for real-time hand tracking
- Extracts hand landmarks and finger positions
- Classifies static ASL letter poses
- Visualizes hand skeleton on video feed

### `ollama_client.py`
- Integrates local Ollama LLM models
- Processes detected emotions/text for contextual responses
- Supports custom prompts and model selection

### `app.py`
- CustomTkinter-based GUI with dark theme
- Purple accent color scheme for modern aesthetics
- Real-time video streaming and display
- Mode switching between Emotion Detection and ASL

## Keyboard Shortcuts

- `Q` or `ESC` - Exit the application
- Mode-specific shortcuts available in the application UI

## Performance Optimization

- Multithreaded video processing to prevent UI blocking
- Configurable frame skip and processing resolution
- Efficient hand landmark calculation with MediaPipe GPU support

## Troubleshooting

### Models Not Loading
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt --upgrade

# Verify hand_landmarker.task exists in project root
ls hand_landmarker.task
```

### Webcam Issues
- Verify camera permissions are granted
- Check if another application is using the webcam
- Try restarting the application

### Ollama Connection Error
- Ensure Ollama is installed and running
- Verify Ollama is accessible at localhost:11434
- Start Ollama service: `ollama serve`

## Future Enhancements

- [ ] Support for continuous ASL sentence recognition
- [ ] Emotion-based response generation
- [ ] Multi-hand gesture support
- [ ] Custom emotion classification training
- [ ] Video recording with annotations
- [ ] Export functionality for analysis


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue in the repository.

---

**Built with ❤️ using Python, OpenCV, and MediaPipe**
