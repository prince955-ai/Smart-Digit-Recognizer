
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.keras")

def load_dataset():
    """Load and normalize the MNIST handwritten digit dataset."""
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # CNN expects image shape: height, width, channels
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Create a Convolutional Neural Network."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(
            32, kernel_size=3, activation="relu"
        ),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(
            64, kernel_size=3, activation="relu"
        ),
        layers.MaxPooling2D(pool_size=2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_dataset()

    print("Training the CNN model...")
    model = build_model()

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )

    model.save(MODEL_PATH)

    results = {
        "test_accuracy": float(accuracy),
        "test_loss": float(loss),
        "epochs": 5,
        "dataset": "MNIST",
        "architecture": "Convolutional Neural Network"
    }

    with open("results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print(f"Test loss: {loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
