```markdown
# Real-Time Detection & Classification Projects

Welcome to the **Real-Time Detection & Classification** repository! This repo contains two distinct but related deep-learning and computer-vision projects:

1. **Real-Time Fire & Accident Detection System**  
2. **Food Image Classification**

Each project is self-contained in its own directory with code, model checkpoints, and documentation. Below you will find a high-level overview, setup instructions, usage examples, and contribution guidelines for both projects.

---

## Table of Contents

1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Real-Time Fire & Accident Detection System](#real-time-fire--accident-detection-system)  
   - [Description](#description-1)  
   - [Dependencies](#dependencies-1)  
   - [Setup & Installation](#setup--installation-1)  
   - [Usage](#usage-1)  
   - [Directory Structure](#directory-structure-1)  
   - [Results & Demo](#results--demo-1)  
4. [Food Image Classification](#food-image-classification)  
   - [Description](#description-2)  
   - [Dependencies](#dependencies-2)  
   - [Setup & Installation](#setup--installation-2)  
   - [Usage](#usage-2)  
   - [Directory Structure](#directory-structure-2)  
   - [Results & Demo](#results--demo-2)  
5. [Contributing](#contributing)  
6. [License](#license)  
7. [Contact](#contact)  

---

## Overview

This repository demonstrates two practical applications of deep learning and computer vision:

1. **Real-Time Fire & Accident Detection System**  
   - A lightweight, CPU-only solution built with Python, OpenCV, and YOLOv8.  
   - Continuously processes live CCTV feed to detect fire or accident events, then triggers alarms and desktop notifications in real time.

2. **Food Image Classification**  
   - A MobileNetV2-based image classifier trained on a comprehensive food dataset.  
   - Accurately recognizes a wide variety of dish/food categories for use in dietary apps, restaurant automation, and culinary-learning platforms.

Each subfolder contains:
- Clean, commented Python scripts  
- Pretrained model checkpoints (when available)  
- Example dataset splits or instructions to download public datasets  
- Clear instructions on how to run training and inference  

---

## Project Structure

```

├── fire\_accident\_detection/
│   ├── models/
│   │   └── yolov8\_fire\_accident.pt
│   ├── src/
│   │   ├── detector.py
│   │   ├── notify.py
│   │   └── utils.py
│   ├── requirements.txt
│   ├── README.md
│   └── demo\_videos/
│       └── sample\_cctv.mp4
│
├── food\_classification/
│   ├── data/
│   │   ├── train/
│   │   └── val/
│   ├── models/
│   │   └── mobilenetv2\_food.pth
│   ├── src/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── LICENSE

```

> **Note:** Each subproject has its own `README.md` containing detailed instructions. The following sections highlight the key points; refer to individual subdirectories for more granular details.

---

## Real-Time Fire & Accident Detection System

### Description

- **Goal**: Detect fires and road accidents from live CCTV footage in real time, trigger a sound alarm, and send a desktop notification.  
- **Core Technologies**:  
  - **YOLOv8** (for object detection)  
  - **Python 3.9+**  
  - **OpenCV** (for video capture and frame processing)  
  - **Plyer** (for desktop notifications)  
  - **SimpleAudio (or playsound)** (for sound alarms)  
- **Features**:  
  1. CPU-only inference using optimized YOLOv8 weights.  
  2. Real-time bounding-box visualization on each frame.  
  3. Sound alarm (WAV/MP3) upon detection.  
  4. Desktop notification popup with custom message (“Fire Detected!” or “Accident Detected!”).  
  5. Configurable confidence threshold and class labels.  
  6. Easily extensible to additional classes or alert channels (e.g., email, SMS, mobile push).

---

### Dependencies

All required Python packages are listed in `fire_accident_detection/requirements.txt`. Key dependencies include:
```

opencv-python
ultralytics            # YOLOv8 implementation
numpy
plyer                  # Cross-platform desktop notifications
simpleaudio (or playsound)  # For sound alerts

````
> **Tip**: It is recommended to use a virtual environment (venv or conda) before installing dependencies.

---

### Setup & Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/Himanshu420247/project.git
   cd project/fire_accident_detection
````

2. **Create and activate a Python virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install Python packages**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify YOLOv8 Model Weights**

   * If `models/yolov8_fire_accident.pt` is provided, no action is needed.
   * Otherwise, train your own weights using sample scripts or download from a shared location.

---

### Usage

1. **Prepare a CCTV feed** (USB camera, IP camera, or a test video):

   * By default, the script captures from `cv2.VideoCapture(0)` (webcam).
   * To use a pre-recorded file, pass its path to the `--source` flag.

2. **Run the real-time detector**:

   ```bash
   python src/detector.py \
     --weights models/yolov8_fire_accident.pt \
     --source 0 \
     --conf-thres 0.25
   ```

   * `--weights`: Path to the YOLOv8 weights file.
   * `--source`: Camera index (e.g., 0 for default webcam) or path to a video file.
   * `--conf-thres`: Minimum confidence threshold for detections (default 0.25).

3. **Notification & Alarm**

   * Upon detecting a “fire” or “accident,” the script will:

     1. Play a short alarm sound (`alarm.wav`).
     2. Send a desktop notification with a custom title and message.
   * See `src/notify.py` for how notifications and sounds are triggered and customized.

4. **Stop the script**

   * Press `q` in the display window or interrupt with `Ctrl+C` in the terminal.

---

### Directory Structure

```
fire_accident_detection/
├── models/
│   └── yolov8_fire_accident.pt        # Pretrained YOLOv8 weights
├── src/
│   ├── detector.py                     # Main real-time detection script
│   ├── notify.py                       # Notification & sound logic
│   └── utils.py                        # Helper functions (e.g., label mapping)
├── demo_videos/
│   └── sample_cctv.mp4                 # Example input video
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

### Results & Demo

* **Live Video Feed**:
  Each frame displays bounding boxes around detected fires or accidents with class labels and confidence scores.

* **Sound Alarm**:
  A distinct audio alert (`alarm.wav`) plays when an incident is detected.

* **Desktop Notification**:
  A system notification pops up instantaneously. Example:

  > **Title**: Fire Detected!
  > **Message**: “Location: Main Gate | Confidence: 0.78”

<details>
<summary>Example Screenshot</summary>

![Fire Detection Demo Screenshot](./demo_videos/fire_demo_screenshot.png)

*Note: Screenshot is for illustration; replace with your own sample.*

</details>

---

## Food Image Classification

### Description

* **Goal**: Build and train a deep-learning model to classify images of food into predefined categories (e.g., Pizza, Salad, Sushi).
* **Core Technologies**:

  * **PyTorch (or TensorFlow / Keras)**
  * **MobileNetV2** (pretrained backbone + transfer learning)
  * **Python 3.9+**
  * **OpenCV** or **Pillow** (for image preprocessing)
  * **Torchvision** (for dataset handling and augmentations)
* **Features**:

  1. Transfer learning on MobileNetV2 backbone.
  2. Data augmentation (rotation, flipping, scaling) to improve generalization.
  3. Training / validation loops with real-time loss/accuracy logging.
  4. Inference script to classify a single image or a batch of images.
  5. Configurable hyperparameters (learning rate, batch size, epochs) via command-line or config file.

---

### Dependencies

All required Python packages are listed in `food_classification/requirements.txt`. Key dependencies include:

```
torch
torchvision
numpy
opencv-python
tqdm
matplotlib  # (optional, for loss/accuracy plots)
```

> **Tip**: Use a GPU-enabled environment (CUDA) if available to speed up training.

---

### Setup & Installation

1. **Clone this repository** (if not already done):

   ```bash
   git clone https://github.com/Himanshu420247/project.git
   cd project/food_classification
   ```

2. **Create and activate a Python virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

3. **Install Python packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**

   * Download a public food image dataset (e.g., Food-101) or use your custom dataset.
   * Organize images into the following folder structure:

     ```
     food_classification/data/
     ├── train/
     │   ├── category_1/
     │   ├── category_2/
     │   └── ...
     └── val/
         ├── category_1/
         ├── category_2/
         └── ...
     ```
   * Ensure each subfolder name matches the class label you want to train on.

---

### Usage

1. **Training**

   ```bash
   python src/train.py \
     --data-dir ./data \
     --backbone mobilenet_v2 \
     --epochs 20 \
     --batch-size 32 \
     --learning-rate 0.001 \
     --output-dir ./models
   ```

   * `--data-dir`: Root folder containing `train/` and `val/` subfolders.
   * `--backbone`: Name of the pretrained model (e.g., `mobilenet_v2`, `resnet50` if available).
   * `--epochs`: Number of training epochs.
   * `--batch-size`: Batch size for training/validation.
   * `--learning-rate`: Initial learning rate.
   * `--output-dir`: Directory to save the best model checkpoint (e.g., `mobilenetv2_food.pth`).

2. **Evaluation**

   ```bash
   python src/evaluate.py \
     --model-path ./models/mobilenetv2_food.pth \
     --data-dir ./data/val
   ```

   * Prints validation accuracy, loss, and a confusion matrix (if implemented).

3. **Inference / Prediction**

   ```bash
   python src/predict.py \
     --model-path ./models/mobilenetv2_food.pth \
     --image-path path/to/your/image.jpg \
     --classes-file classes.txt
   ```

   * `--classes-file`: A text file listing class names in the same order as training labels.
   * Outputs the predicted class label and confidence score.

---

### Directory Structure

```
food_classification/
├── data/
│   ├── train/                      # Training images (organized by category)
│   └── val/                        # Validation images (organized by category)
├── models/
│   └── mobilenetv2_food.pth        # Best model checkpoint after training
├── src/
│   ├── train.py                    # Training script with transfer learning
│   ├── evaluate.py                 # Validation and metrics script
│   └── predict.py                  # Single-image inference script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

### Results & Demo

* After training for **20 epochs** on a subset of Food-101 (or a custom dataset), the model achieves:

  * **Validation Accuracy**: \~85–90% (depending on dataset split & hyperparameters)
  * **Loss Curve**: Demonstrates stable convergence within 10–15 epochs

<details>
<summary>Example Predictions</summary>

| Input Image                       | Predicted Label | Confidence |
| --------------------------------- | --------------- | ---------- |
| ![pizza](./demo_images/pizza.jpg) | Pizza           | 0.93       |
| ![sushi](./demo_images/sushi.jpg) | Sushi           | 0.89       |
| ![salad](./demo_images/salad.jpg) | Salad           | 0.91       |

*Note: Replace `./demo_images/…` with your own example images folder.*

</details>

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.
2. **Create** a new feature branch:

   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Commit** your changes with a descriptive message:

   ```bash
   git commit -m "Add <feature/bugfix description>"
   ```
4. **Push** your branch to your fork:

   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open** a Pull Request on the main repository.
6. We will review your changes, request revisions if needed, and merge once approved.

Feel free to file issues for bugs or feature requests. Please provide relevant details, code snippets, and screenshots where applicable.

---

## License

This repository is released under the **MIT License**. See [LICENSE](./LICENSE) for full details.

---

## Contact

Your Name · [@YourTwitterHandle](https://twitter.com/YourTwitterHandle) · [your.email@example.com](mailto:your.email@example.com)
Project Link: [https://github.com/Himanshu420247/project](https://github.com/Himanshu420247/project)

Thank you for checking out this repository! We hope it helps you build and extend real-time detection and classification systems for your own applications.
