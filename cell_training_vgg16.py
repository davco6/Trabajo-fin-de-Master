"""
Modificación del script desarrollado por Abner Ayala-Acevedo
https://www.kaggle.com/abnera/transfer-learning-keras-xception-cnn

Estructura del directorio de trabajo:
          1000 imágenes de restos celulares situados en la carpeta cell_debris
          1000 imágenes de imágenes con un solo núcleo
          1000 imágenes de imágenes con conglomerados de núcleos
          
          
data/
    train/
        cell_debris/
            001.jpg
            002.jpg
            ...
        I/
            001.jpg
            002.jpg
            ...
        II/
            001.jpg
            002.jpg
            ...
    validation/
        cell_debris/
            001.jpg
            002.jpg
            ...
        I/
            001.jpg
            002.jpg
            ...
        II/
            001.jpg
            002.jpg
            ...

"""
import numpy as np
import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as k



top_model_weights_path = 'bottleneck_fc_model.h5'
# Dimensiones de la imagen
img_width, img_height = 192, 192

nb_classes = 3  # Número de clases
based_model_last_block_layer_number = 15  #Bloqueo del entrenamiento de las capas. Para el modelo VGG16 en el que se entrenan sólo el último modulo el valor es 15.

batch_size = 32  
nb_epoch = 100


def train(train_data_dir, validation_data_dir, model_path):
    # pre-entrenamiento del modelo VGG16
    base_model = VGG16(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

    # Bloqueo del las capas de más arriba
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)
    print(model.summary())

    #Se congelan todas la capas del modelo
    for layer in base_model.layers:
        layer.trainable = False

    #Entrenamiento de la base de datos de imágenes
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       rotation_range=30,
                                       shear_range=0.2,
                                       zoom_range=0.5,
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       fill_mode="nearest",
                                       vertical_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    
    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
    

    callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=10, verbose=0),
        CSVLogger('bottleneck_vgg16.log')
    ]

    # Entenamiento del cuello de botella
    model.fit_generator(train_generator,
                        steps_per_epoch=3000/batch_size,
                        nb_epoch=nb_epoch / 5,
                        validation_data=validation_generator,
                        nb_val_samples=210/batch_size,
                        callbacks=callbacks_list)

    
    print("\nStarting to Fine Tune Model\n")

    
    # Cargamos los "mejores" pesos obtenidos mediante el entrenamiento del cuello de botella
    model.load_weights(top_weights_path)

    # Para el entrenamiento del último módulo de la arquitectura VGG16 congelamos todas las capas convolucionales y de pooling menos las del último módulo.
    for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True

    # Compilamos el modelo con una menor velocidad de aprendizaje
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    
    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
    callbacks_list = [
        ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=20, verbose=0),
        CSVLogger('finetuning.log')
    ]

    
    model.fit_generator(train_generator,
                        steps_per_epoch=3000/batch_size,
                        nb_epoch=nb_epoch,
                        validation_data=validation_generator,
                        nb_val_samples=210/batch_size,
                        callbacks=callbacks_list)

    
    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print('Los argumentos deben estructurarse como:\npython code/train.py <data_dir/> <model_dir/>')
        print('Ejemplo: python code/train.py data/cells/ model/cells/')
        sys.exit(2)
    else:
        data_dir = os.path.abspath(sys.argv[1])
        train_dir = os.path.join(os.path.abspath(data_dir), 'train') 
        validation_dir = os.path.join(os.path.abspath(data_dir), 'validation')  
        model_dir = os.path.abspath(sys.argv[2])

        os.makedirs(os.path.join(os.path.abspath(data_dir), 'preview'), exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    train(train_dir, validation_dir, model_dir) 
    k.clear_session()
