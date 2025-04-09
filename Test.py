
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models



# Load and prepare the MNIST dataset
print("Loading MNIST dataset...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape images to add a channel 
# dimension (required for Conv2D)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Build the model
print("Building CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
if input("Press Enter to start training the AI model: "):
    print("Training the vision AI to detect numbers...")
    epochs = 5
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        history = model.fit(
            train_images, train_labels, 
            epochs=1,
            validation_data=(test_images, test_labels),
            verbose=1
        )
        
        # After each epoch, let user test some samples
        if epoch < epochs-1 and input("\nPress Enter to see some test results or 'q' to continue training: ") != 'q':
            # Test the model with a few examples
            for i in range(3):
                test_idx = np.random.randint(0, len(test_images))
                
                # Make prediction
                img = test_images[test_idx:test_idx+1]
                prediction = model.predict(img, verbose=0)
                predicted_digit = np.argmax(prediction[0])
                
                # Display the image
                plt.figure(figsize=(4, 4))
                plt.imshow(test_images[test_idx].reshape(28, 28), cmap='gray')
                plt.title(f'Actual: {test_labels[test_idx]}, Predicted: {predicted_digit}')
                plt.show()
                
                # Continue or quit testing
                if i < 2 and input("Press Enter to see another test or 'q' to continue: ") == 'q':
                    break
    
    # Final evaluation
    print("\nEvaluating the model...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Save the model
    if input("\nPress Enter to save the model or 'n' to skip: ").lower() != 'n':
        model.save('number_recognition_model.h5')
        print("Model saved as 'number_recognition_model.h5'")
    
    print("\nEnd of the AI training session")