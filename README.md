# 🌸 Flower Recognition using EfficientNet-B0

A complete **deep learning image classification project** that
recognizes flowers from an uploaded image using **EfficientNet-B0**,
**PyTorch**, **FastAPI**, and **Streamlit**.

The project is built as an end-to-end application:

``` text
                    🌸 FLOWER IMAGE
                           │
                           ▼
                    🖥️ Streamlit
                      Frontend
                           │
                     HTTP POST
                           │
                           ▼
                    ⚡ FastAPI
                      Backend
                           │
                           ▼
                  🧠 EfficientNet-B0
                           │
                           ▼
                     🎯 Prediction
                           │
                    ┌──────┴──────┐
                    ▼             ▼
               Flower Name    Confidence
```

------------------------------------------------------------------------

## 🌸 Application Preview

The Streamlit application allows the user to upload a flower image and
receive a prediction with its confidence score.

### Example

``` text
🌸 Flower Recognition

Upload a flower image

[ Choose File ]

e.jpg
436.4 KB

### Uploaded Image

[        🌸 Flower Image        ]

🌸 Prediction: rose

Confidence
73.24%
```

> **Note:** The `73.24%` value above is an example prediction result
> from the application. It is not a claim that every rose image will
> receive this confidence.

------------------------------------------------------------------------

## 🚀 Features

-   🌸 Flower image classification
-   🧠 EfficientNet-B0 transfer learning
-   🔥 PyTorch-based deep learning model
-   🎮 CUDA/GPU support
-   ⚡ FastAPI REST backend
-   🖥️ Streamlit web frontend
-   📤 Image upload through the browser
-   🖼️ Displays the uploaded image
-   🎯 Returns predicted flower class
-   📊 Returns prediction confidence
-   📚 Interactive FastAPI Swagger documentation
-   🧪 Separate training and inference workflow
-   📦 Python virtual environment support

------------------------------------------------------------------------

## 🌺 Supported Flower Classes

The current model is configured for **5 flower classes**:

``` text
1. Daisy
2. Dandelion
3. Rose
4. Sunflower
5. Tulip
```

The class order used by the backend is:

``` python
class_names = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]
```

Make sure this order matches the class mapping used while training the
model.

------------------------------------------------------------------------

# 🧠 Model

## EfficientNet-B0

The project uses **EfficientNet-B0** as the image classification model.

Instead of training an entire convolutional neural network from the
beginning, the project uses **transfer learning**.

The original EfficientNet-B0 classifier is replaced with a classifier
containing 5 output neurons:

``` python
model = efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)
```

The five output neurons correspond to the five flower classes.

------------------------------------------------------------------------

# 🔄 Image Processing Pipeline

Before an image is passed to EfficientNet-B0, it is transformed into the
format expected by the model.

``` python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Pipeline

``` text
Original Image
      │
      ▼
Resize → 224 × 224
      │
      ▼
Convert to Tensor
      │
      ▼
Normalize
      │
      ▼
EfficientNet-B0
      │
      ▼
5 Class Probabilities
      │
      ▼
Predicted Flower
```

------------------------------------------------------------------------

# 🏋️ Training

The model was trained using PyTorch.

The training process included:

-   Image preprocessing
-   Dataset splitting
-   DataLoader
-   Transfer learning
-   Layer freezing
-   Layer unfreezing/fine-tuning
-   Cross Entropy Loss
-   Adam/SGD optimization
-   Validation
-   Early stopping
-   Accuracy and loss tracking
-   GPU acceleration

## Fine-Tuning

Initially, the pretrained network layers can be frozen:

``` python
for param in model.parameters():
    param.requires_grad = False
```

Then selected later layers can be unfrozen for fine-tuning:

``` python
for param in model.features[-2:].parameters():
    param.requires_grad = True
```

The classifier remains trainable:

``` python
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)
```

This allows the pretrained network to retain useful visual features
while adapting its deeper features to the flower dataset.

------------------------------------------------------------------------

# 🛑 Early Stopping

Early stopping was used to prevent unnecessary training once validation
performance stopped improving.

Example:

``` text
Epoch [14/500] | Train Loss: 0.2399 | Val Loss: 0.2873 |
Train Acc: 91.91% | Val Acc: 89.35%

EarlyStopping Counter: 5/5

Epoch [15/500] | Train Loss: 0.2277 | Val Loss: 0.2905 |
Train Acc: 92.46% | Val Acc: 89.70%

Early Stopping Triggered!
```

This prevents the model from continuing to train when validation loss is
no longer improving.

------------------------------------------------------------------------

# 📁 Project Structure

``` text
01_flower_recognise/
│
├── backend/
│   ├── main.py
│   ├── best_model.pth
│   └── __pycache__/
│
├── frontend/
│   └── app.py
│
├── flowers/
│   └── raw flower dataset
│
├── Output/
│   ├── train/
│   └── val/
│
├── Output_two/
│   ├── train/
│   └── val/
│
├── Flowers.ipynb
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# ⚡ Backend

The backend is built using **FastAPI**.

The backend:

1.  Receives an image.
2.  Reads the image.
3.  Converts it to RGB.
4.  Resizes it to `224 × 224`.
5.  Applies normalization.
6.  Sends it to EfficientNet-B0.
7.  Calculates probabilities using Softmax.
8.  Selects the class with the highest probability.
9.  Returns the flower name and confidence.

## API Endpoint

``` text
POST /predict
```

Example response:

``` json
{
    "filename": "e.jpg",
    "prediction": "rose",
    "confidence": 73.24
}
```

------------------------------------------------------------------------

# 📚 FastAPI Documentation

FastAPI automatically provides an interactive API testing interface.

Start the backend:

``` bash
uvicorn backend.main:app --reload
```

Then open:

``` text
http://127.0.0.1:8000/docs
```

From the Swagger interface you can:

-   Select an image
-   Send the image to the API
-   Run EfficientNet-B0 inference
-   View the JSON response

------------------------------------------------------------------------

# 🖥️ Frontend

The frontend is built using **Streamlit**.

The user can:

1.  Open the website.
2.  Upload an image.
3.  View the uploaded image.
4.  Click **Predict Flower**.
5.  Streamlit sends the image to FastAPI.
6.  FastAPI performs inference.
7.  Streamlit displays the prediction and confidence.

Start Streamlit with:

``` bash
streamlit run frontend/app.py
```

The application will normally be available at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# 🔗 Frontend → Backend Communication

The Streamlit frontend communicates with the FastAPI backend using an
HTTP POST request.

``` python
response = requests.post(
    API_URL,
    files={
        "file": (
            uploaded_file.name,
            uploaded_file,
            uploaded_file.type
        )
    }
)
```

The backend URL is:

``` python
API_URL = "http://127.0.0.1:8000/predict"
```

------------------------------------------------------------------------

# 💻 Installation

## 1. Clone or copy the project

Open a terminal in the project directory.

``` bash
cd 01_flower_recognise
```

------------------------------------------------------------------------

## 2. Activate the virtual environment

The project uses a Python virtual environment.

Example on Windows PowerShell:

``` powershell
& "C:\Users\Asus\pytorch_env\Scripts\Activate.ps1"
```

After activation you should see:

``` text
(pytorch_env)
```

in the terminal.

------------------------------------------------------------------------

## 3. Install dependencies

``` bash
pip install -r requirements.txt
```

The project requires packages such as:

``` text
torch
torchvision
numpy
pandas
Pillow
scikit-learn
fastapi
uvicorn
python-multipart
streamlit
requests
```

------------------------------------------------------------------------

# 🎮 GPU Support

The model supports CUDA when a compatible NVIDIA GPU and CUDA-enabled
PyTorch installation are available.

The backend automatically chooses the GPU when available:

``` python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Otherwise it falls back to CPU.

Example:

``` text
Using device: cuda
Model loaded successfully!
```

------------------------------------------------------------------------

# 🧪 Running the Complete Application

You need two terminals.

## Terminal 1 --- FastAPI

Activate the environment and run:

``` bash
uvicorn backend.main:app --reload
```

You should see:

``` text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

------------------------------------------------------------------------

## Terminal 2 --- Streamlit

Activate the same environment and run:

``` bash
streamlit run frontend/app.py
```

Then open:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# 🔬 Complete Prediction Flow

``` text
                 👤 User
                   │
                   ▼
            📤 Upload Image
                   │
                   ▼
             🖥️ Streamlit
                   │
                   │ HTTP POST
                   ▼
             ⚡ FastAPI
                   │
                   ▼
            🖼️ PIL Image
                   │
                   ▼
          Resize 224 × 224
                   │
                   ▼
              Normalize
                   │
                   ▼
          🧠 EfficientNet-B0
                   │
                   ▼
             Softmax
                   │
                   ▼
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
 🌸 Flower Class          📊 Confidence
       │                       │
       └───────────┬───────────┘
                   ▼
              🖥️ Streamlit
                   │
                   ▼
              👤 User
```

------------------------------------------------------------------------

# 📊 Example Prediction

### Input

``` text
e.jpg
436.4 KB
```

### Output

``` text
🌸 Prediction: rose

Confidence
73.24%
```

The uploaded image is displayed on the Streamlit page before the
prediction result.

------------------------------------------------------------------------

# ⚠️ Important Limitation: Unknown Images

The current 5-class classifier has an important limitation.

If an image does not belong to one of the five supported classes, a
normal softmax classifier will still choose one of the five classes.

For example:

``` text
Input: 🐶 Dog

Possible output:
Prediction: rose
Confidence: 85%
```

This does **not** mean the dog is actually a rose.

It happens because the classifier is designed to choose among the five
known classes.

Therefore, confidence from softmax alone should not be treated as a
reliable "is this one of my classes?" detector.

A future version of this project can implement **Out-of-Distribution
(OOD) detection** so that unsupported images can be returned as:

``` text
❌ Unknown

This image does not belong to the supported flower categories.
```

------------------------------------------------------------------------

# 🚀 Future Improvements

The project can be extended with:

-   ❌ Unknown/OOD image detection
-   📊 Confusion matrix
-   📈 Precision, Recall and F1-score
-   🧪 Test-set evaluation
-   🌺 More flower classes
-   🔍 Top-3 predictions
-   📊 Prediction probability chart
-   🖼️ Better image preprocessing
-   🔄 Data augmentation
-   🎯 Better threshold calibration
-   🧠 Fine-tuning more EfficientNet layers
-   📱 Mobile application
-   ☁️ Cloud deployment
-   🐳 Docker deployment
-   🔐 Production API configuration
-   📝 Prediction history
-   📷 Camera-based prediction

------------------------------------------------------------------------

# 🧰 Technologies Used

  Technology        Purpose
  ----------------- ---------------------------------------
  Python            Main programming language
  PyTorch           Deep learning framework
  Torchvision       Computer vision models and transforms
  EfficientNet-B0   Image classification model
  Pillow            Image processing
  NumPy             Numerical operations
  Pandas            Data handling
  Scikit-learn      Evaluation utilities
  FastAPI           Backend REST API
  Uvicorn           FastAPI server
  Streamlit         Web frontend
  Requests          Frontend → backend communication
  CUDA              GPU acceleration

------------------------------------------------------------------------

# 🎓 Concepts Practiced

This project is useful for practicing several important CNN/deep
learning concepts:

### Computer Vision

-   Image loading
-   Image resizing
-   RGB conversion
-   Tensor conversion
-   Normalization
-   Image classification

### CNN / Deep Learning

-   Convolutional neural networks
-   Transfer learning
-   EfficientNet architecture
-   Feature extraction
-   Freezing layers
-   Unfreezing layers
-   Fine-tuning
-   Softmax
-   Cross-entropy loss
-   Optimizers
-   Batch training
-   GPU training

### Model Evaluation

-   Training loss
-   Validation loss
-   Training accuracy
-   Validation accuracy
-   Confusion matrix
-   Precision
-   Recall
-   F1-score

### Deployment

-   REST API
-   FastAPI
-   Streamlit
-   HTTP requests
-   Image upload
-   Model inference
-   Frontend/backend architecture

------------------------------------------------------------------------

# 📌 Project Status

``` text
✅ Dataset preparation
✅ CNN image classification
✅ EfficientNet-B0
✅ Transfer learning
✅ Layer freezing
✅ Fine-tuning
✅ Early stopping
✅ Model saving/loading
✅ FastAPI backend
✅ Streamlit frontend
✅ Image upload
✅ Flower prediction
✅ Confidence score

🔄 Unknown/OOD detection
🔄 Advanced evaluation
🔄 Production deployment
```

------------------------------------------------------------------------

# 🌸 Final Result

This project demonstrates an end-to-end computer vision pipeline:

``` text
Dataset
   ↓
Image Preprocessing
   ↓
EfficientNet-B0
   ↓
Transfer Learning
   ↓
Fine-Tuning
   ↓
Model Evaluation
   ↓
Saved PyTorch Model
   ↓
FastAPI Backend
   ↓
Streamlit Frontend
   ↓
🌸 Flower Prediction
```

The project is designed not only as a flower classifier but also as a
practical way to learn how a **deep learning model is taken from
training to a usable web application**.

------------------------------------------------------------------------

## 👨‍💻 Author

**Saurabh Agrawal**

MCA --- Artificial Intelligence & Machine Learning

------------------------------------------------------------------------

## ⭐ If you find this project useful

Consider improving it further with:

``` text
OOD Detection
      +
Better Evaluation
      +
More Classes
      +
Cloud Deployment
```

to turn it into a more complete computer vision application.
