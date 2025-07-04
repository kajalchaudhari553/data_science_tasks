import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

def train_and_predict_cifar10():
    print(" Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print(" Normalizing data...")
    X_train, X_test = X_train / 255.0, X_test / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print("Building CNN model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)  # No softmax here (from_logits=True)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("Training model...")
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\n Test Accuracy: {test_acc:.2f}')

    # Plot training and validation accuracy/loss
    print(" Plotting training history...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show(block=True)

    # Predict the first 9 images
    print(" Making predictions...")
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    predictions = probability_model.predict(X_test)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_test[i])
        predicted_label = class_names[np.argmax(predictions[i])]
        true_label = class_names[y_test[i][0]]
        color = 'blue' if predicted_label == true_label else 'red'
        plt.xlabel(f"{predicted_label} ({true_label})", color=color)

    plt.tight_layout()
    plt.show(block=True)

    print("Prediction visualization completed.")
    return test_acc, predictions


# Call the function and capture return values
if __name__ == "__main__":
    accuracy, preds = train_and_predict_cifar10()
    print("Returned Test Accuracy:", accuracy)
    print("Returned Predictions Shape:", preds.shape)
