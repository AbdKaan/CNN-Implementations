from keras.layers import Conv2D, Dense, Flatten, Activation, AveragePooling2D, MaxPooling2D, Activation, ZeroPadding2D
from keras.engine.input_layer import Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.merge import add
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split

def first_stage(x):
    x = ZeroPadding2D((3, 3))(x)
    x = Conv2D(filters=64, kernel_size=7, strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    return x

def residual_block(x, filters, first_layer = False, bottleneck = True):
    # Shortcut Connection
    if first_layer:
        x_shortcut = Conv2D(filters=filters*4, kernel_size=1, strides=2, kernel_initializer='he_normal')(x)
        x_shortcut = BatchNormalization()(x_shortcut)
    else:
        x_shortcut = x

    # Main Connection
    if bottleneck:
        if first_layer:
            x = Conv2D(filters=filters, kernel_size=1, strides=2, kernel_initializer='he_normal')(x)
        else:
            x = Conv2D(filters=filters, kernel_size=1, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=filters, kernel_size=3, padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=filters*4, kernel_size=1, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
    else:
        if first_layer:
            x = Conv2D(filters=filters, kernel_size=3, strides=2, padding="same", kernel_initializer='he_normal')(x)
        else:
            x = Conv2D(filters=filters, kernel_size=3, padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=filters, kernel_size=3, padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

    # Add connections.
    x = add([x, x_shortcut])
    x = Activation("relu")(x)

    return x

def output_stage(x, output_size):
    x = AveragePooling2D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(output_size)(x)
    x = Activation("softmax")(x)
    
    return x

def ResNet(input_shape=(224, 224, 3), layer_size=152, output_size=1000):
    # If you would like to try another layer/block sizes, add it in this dictionary and change the layer_size parameter when initializing ResNet.
    blocks_for_stages = {18: [2, 2, 2, 2],
             34: [3, 4, 6, 3],
             50: [3, 4, 6, 3],
             101: [3, 4, 23, 3],
             152: [3, 8, 36, 3]}

    x_input = Input(input_shape)
    x = first_stage(x_input)

    bottleneck = layer_size in [101, 152]
    filters = 32

    for stage in range(len(blocks_for_stages[layer_size])):
        filters *= 2
        for block in range(blocks_for_stages[layer_size][stage]):
            if block is 0:
                x = residual_block(x, filters, first_layer=True, bottleneck=bottleneck)
            else:
                x = residual_block(x, filters, bottleneck=bottleneck)

    x = output_stage(x, output_size)

    model = Model(inputs = x_input, outputs = x, name='ResNet')
    return model

def load_and_transform_data(test_size=0.20):
    """
    Returns oxflower17 dataset by splitting them to train and test sets.
    """
    X, Y = oxflower17.load_data(one_hot=True)

    print("X's shape: ", X.shape)
    print("Y's shape: ", Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    print("X_train's shape: ", X_train.shape)
    print("Y_train's shape: ", Y_train.shape)

    return X_train, X_test, Y_train, Y_test

def train_and_test_model(X_train, X_test, Y_train, Y_test, layer_size, class_size):
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001)
    model = ResNet(X_train.shape[1:], layer_size, class_size)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_split=0.25, shuffle=True)

    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=256)
    print("Loss on test set: ", loss_and_metrics[0])
    print("Accuracy on test set: ", loss_and_metrics[1])

X_train, X_test, Y_train, Y_test = load_and_transform_data()
train_and_test_model(X_train, X_test, Y_train, Y_test, 152, Y_train.shape[1])
