
import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(img_h: int, img_w: int):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_h, img_w, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train', help='root with /player and /background subfolders')
    parser.add_argument('--img_w', type=int, default=64)
    parser.add_argument('--img_h', type=int, default=128)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--out', default='models/player_detector.h5')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    datagen = ImageDataGenerator(
        rescale=1.0/255.0, validation_split=0.2,
        rotation_range=8, width_shift_range=0.05, height_shift_range=0.05,
        shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode='nearest'
    )

    train_data = datagen.flow_from_directory(
        args.data, target_size=(args.img_h, args.img_w),
        batch_size=args.batch, class_mode='binary', subset='training'
    )

    val_data = datagen.flow_from_directory(
        args.data, target_size=(args.img_h, args.img_w),
        batch_size=args.batch, class_mode='binary', subset='validation'
    )

    model = build_model(args.img_h, args.img_w)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor='val_accuracy', mode='max'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', mode='max')
    ]

    model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)

    # Final save (ensures a model file exists even if not 'best')
    model.save(args.out)
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main()
