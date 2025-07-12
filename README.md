
# 🤖 Deep Learning with TensorFlow

This project demonstrates **basic deep learning techniques using TensorFlow and Keras**. It includes hands-on code for building, training, and evaluating a neural network to recognize handwritten digits from the **MNIST** dataset.

---

## 📘 Project Overview

This notebook is designed for beginners who are learning **TensorFlow** and **Deep Learning**. It covers:

- Loading and preparing datasets (MNIST)
- Visualizing image data
- Creating neural network models
- Compiling, training, and evaluating models
- Making predictions on test data
- Understanding model performance

---

## 🧠 Concepts Covered

- Feedforward Neural Networks
- Activation Functions (ReLU, Softmax)
- One-hot Encoding
- Training/Validation Split
- Model Evaluation Metrics (Loss, Accuracy)
- Prediction Visualization

---

## 🔧 Setup Instructions

1. ✅ **Install TensorFlow (Python 3.10 recommended):**

```bash
pip install tensorflow
````

> **⚠️ Important**: TensorFlow is **not compatible with Python 3.12** as of now.

2. ✅ **Run the notebook**:

```bash
jupyter notebook
# or
streamlit run app.py  # if you create a frontend
```

---

## 📦 Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
```

---

## 📊 Data Preparation

```python
# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize images to range [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# One-hot encode labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)
```

---

## 🧱 Model Building

```python
# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
```

---

## 🏋️ Training the Model

```python
history = model.fit(X_train, y_train_encoded,
                    validation_split=0.2,
                    epochs=10,
                    batch_size=32)
```

---

## 🧪 Model Evaluation

```python
# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print("Test Accuracy:", test_acc)
```

---

## 🔍 Make Predictions

```python
# Predict on test images
predictions = model.predict(X_test)

# View predictions
plt.imshow(X_test[0], cmap="gray")
plt.title(f"Predicted: {np.argmax(predictions[0])}, Actual: {y_test[0]}")
plt.axis("off")
plt.show()
```

---

## 📈 Training History Plot

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 📌 Conclusion

This project provided a foundational understanding of how to:

* Prepare image data for deep learning
* Build and compile a neural network
* Train and evaluate performance
* Interpret predictions using MNIST dataset

---

## 🚀 Next Steps

* Try Convolutional Neural Networks (CNNs) for better accuracy
* Experiment with dropout layers to reduce overfitting
* Deploy the model using Streamlit or Flask

---

## 🙋‍♂️ Author

**Akshay Bhujbal**

* 📍 Pune, India
* 🧑‍💻 Data Analyst | Aspiring AI Engineer
* 🔗 [GitHub](https://github.com/AkshayBhujbal1995)
* 💼 [LinkedIn](https://linkedin.com/in/akshay-1995-bhujbal)

---

## ⭐ Like this Project?

Please ⭐ star this repo and share with others if you found it helpful!

---
