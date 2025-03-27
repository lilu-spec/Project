from tensorflow import keras

def getData(x):
    pro_x = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(x)
    pro_x = keras.layers.BatchNormalization()(pro_x)
    pro_x = keras.layers.Activation(activation='relu')(pro_x)

    pro_x = keras.layers.Conv1D(filters=256, kernel_size=8, padding='same')(pro_x)
    pro_x = keras.layers.BatchNormalization()(pro_x)
    pro_x = keras.layers.Activation(activation='relu')(pro_x)

    pro_x = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(pro_x)
    pro_x = keras.layers.BatchNormalization()(pro_x)
    pro_x = keras.layers.Activation(activation='relu')(pro_x)

    pro_x = keras.layers.GlobalAveragePooling1D()(pro_x)
    return pro_x
