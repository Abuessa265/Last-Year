import scipy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory('Data', target_size=(224, 224), batch_size=32, class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory('Data', target_size=(224, 224), batch_size=32, class_mode='categorical')

# Load the pre-trained MobileNet model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False
num_classes=10
# Add custom classification layers
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Compile the model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, validation_data=val_data)

# Save the model
#model.save('Model')


# Save the model as an HDF5 file
model.save('model2.h5', save_format='h5')


# Get the class names from the ImageDataGenerator
class_names = list(train_data.class_indices.keys())

# Save the class names to a file
with open('labels2.txt', 'w') as f:
    for class_name in class_names:
        f.write(class_name + '\n')
