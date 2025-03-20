import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from sklearn.preprocessing import OneHotEncoder
from adapt.parameter_based import FineTuning
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from sklearn.manifold import TSNE

import ssl

# bypass the use of certificates
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
ssl._create_default_https_context = ssl._create_unverified_context

resnet50 = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling="avg")
########

EVALUATE = False
USE_DA = False

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--data_root', type=str, help="Path to data", required=False, default="data/office/")
parser.add_argument('--base_model_root', type=str, help="Path to base model", required=False, default="models/base")
parser.add_argument('--epochs', type=str, help="Number of epochs", required=False, default=1)
parser.add_argument('--batch_size', type=str, help="Batch size", required=False, default=12)

args = parser.parse_args()

# Check if GPUs are available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available. Using GPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first available GPU
else:
    print("GPU is not available. Using CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage
data_root = args.data_root

SAVE_IMG_FLAG = False

"""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(plt.imread(data_root+"amazon/images/back_pack/frame_0001.jpg"))
ax2.imshow(plt.imread(data_root+"amazon/images/bike/frame_0001.jpg"))
ax1.set_title("label = back-pack"); ax2.set_title("label = bike");
plt.suptitle("Domain = Amazon")
plt.savefig("img_ex_backpack.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(plt.imread(data_root+"webcam/images/back_pack/frame_0001.jpg"))
ax2.imshow(plt.imread(data_root+"webcam/images/bike/frame_0001.jpg"))
ax1.set_title("label = back-pack"); ax2.set_title("label = bike");
plt.suptitle("Domain = Webcam")
plt.savefig("img_ex_bike.png")
"""

def get_Xy(domain, path_to_folder="office/office_nopreprocessing/"):

    path = path_to_folder + domain +"/images/"
    X = []
    y = []

    for r, d, f in os.walk(path):
        for direct in d:
            if not ".ipynb_checkpoints" in direct:
                for r, d, f in os.walk(os.path.join(path , direct)):
                    for file in f:
                        path_to_image = os.path.join(r, file)
                        if not ".ipynb_checkpoints" in path_to_image:
                            image = Image.open(path_to_image)
                            image = image.resize((224, 224), Image.Resampling.LANCZOS)
                            image = np.array(image, dtype=int)
                            X.append(image)
                            y.append(direct)
    return X, y

"""
Xs, ys = get_Xy("amazon", path_to_folder=data_root)
Xt, yt = get_Xy("webcam", path_to_folder=data_root)

resnet50 = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling="avg")

first_layer = resnet50.get_layer('conv5_block2_out')
#inputs = Input(first_layer.output_shape[1:])
inputs = Input(first_layer.output.shape[1:])

for layer in resnet50.layers[resnet50.layers.index(first_layer)+1:]:
    if layer.name == "conv5_block3_1_conv":
        x = layer(inputs)
    elif layer.name == "conv5_block3_add":
        x = layer([inputs, x])
    else:
        x = layer(x)

first_blocks = Model(resnet50.input, first_layer.output)
last_block = Model(inputs, x)
"""
def load_resnet50(path="resnet50_last_block.hdf5"):
    model = load_model(path)
    for i in range(len(model.layers)):
        if model.layers[i].__class__.__name__ == "BatchNormalization":
            model.layers[i].trainable = False
    return model
"""
last_block.summary()
last_block.save("resnet50_last_block.hdf5")


Xs = first_blocks.predict(preprocess_input(np.stack(Xs)))
Xt = first_blocks.predict(preprocess_input(np.stack(Xt)))

#one = OneHotEncoder(sparse=False)
one = OneHotEncoder(sparse_output=False)

one.fit(np.array(ys).reshape(-1, 1))

ys_lab = one.transform(np.array(ys).reshape(-1, 1))
yt_lab = one.transform(np.array(yt).reshape(-1, 1))

print("X source shape: %s"%str(Xs.shape))
print("X target shape: %s"%str(Xt.shape))
"""

def preprocess_data(X, y):

    resnet50 = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling="avg")

    first_layer = resnet50.get_layer('conv5_block2_out')
    #inputs = Input(first_layer.output_shape[1:])
    inputs = Input(first_layer.output.shape[1:])

    for layer in resnet50.layers[resnet50.layers.index(first_layer)+1:]:
        if layer.name == "conv5_block3_1_conv":
            x = layer(inputs)
        elif layer.name == "conv5_block3_add":
            x = layer([inputs, x])
        else:
            x = layer(x)

    first_blocks = Model(resnet50.input, first_layer.output)
    last_block = Model(inputs, x)

    #last_block.summary()
    #last_block.save("resnet50_last_block.hdf5")


    X = first_blocks.predict(preprocess_input(np.stack(X)))

    #one = OneHotEncoder(sparse=False)
    one = OneHotEncoder(sparse_output=False)

    one.fit(np.array(y).reshape(-1, 1))

    y = one.transform(np.array(y).reshape(-1, 1))

    return X, y

def get_task(dropout=0.5, max_norm=0.5):
    model = Sequential()
    model.add(Dense(1024, activation="relu",
                   kernel_constraint=MaxNorm(max_norm),
                   bias_constraint=MaxNorm(max_norm)))
    model.add(Dropout(dropout))
    model.add(Dense(1024, activation="relu",
                   kernel_constraint=MaxNorm(max_norm),
                   bias_constraint=MaxNorm(max_norm)))
    model.add(Dropout(dropout))
    model.add(Dense(31, activation="softmax",
                   kernel_constraint=MaxNorm(max_norm),
                   bias_constraint=MaxNorm(max_norm)))
    return model

class MyDecay(LearningRateSchedule):

    def __init__(self, max_steps=1000, mu_0=0.01, alpha=10, beta=0.75):
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta
        self.max_steps = float(max_steps)

    def __call__(self, step):
        p = step / self.max_steps
        return self.mu_0 / (1+self.alpha * p)**self.beta
    

def train(Xs, ys, Xt, yt):
    
    lr = 0.04
    momentum = 0.9
    alpha = 0.0002

    optimizer_task = SGD(learning_rate=MyDecay(mu_0=lr, alpha=alpha),
                        momentum=momentum, nesterov=True)
    optimizer_enc = SGD(learning_rate=MyDecay(mu_0=lr/10., alpha=alpha),
                        momentum=momentum, nesterov=True)


    finetunig = FineTuning(
        encoder=load_resnet50(path=os.path.join(args.base_model_root, "resnet50_last_block.hdf5")),
        task=get_task(),
        optimizer=optimizer_task,
        optimizer_enc=optimizer_enc,
        loss="categorical_crossentropy",
        metrics=["acc"],
        copy=False,
        pretrain=True,
        pretrain__epochs=5)
    

    Xs, ys_lab = preprocess_data(Xs, ys)
    Xt, yt_lab = preprocess_data(Xt, yt)

    finetunig.fit(Xs, ys_lab, epochs=args.epochs, batch_size=args.batch_size, validation_data=(Xt, yt_lab))


    acc = finetunig.history.history["acc"]; val_acc = finetunig.history.history["val_acc"]

    plt.plot(acc, label="Train acc - final value: %.3f"%acc[-1])
    plt.plot(val_acc, label="Test acc - final value: %.3f"%val_acc[-1])
    plt.legend(); plt.xlabel("Epochs"); plt.ylabel("Acc")
    if SAVE_IMG_FLAG:
        plt.savefig("train_curve_wo_da.png")
    else:
        plt.show()

    Xs_enc = finetunig.transform(Xs)
    Xt_enc = finetunig.transform(Xt)

    np.random.seed(0)
    X_ = np.concatenate((Xs_enc, Xt_enc))
    X_tsne = TSNE(2).fit_transform(X_)
    plt.figure(figsize=(8, 6))
    plt.plot(X_tsne[:len(Xs), 0], X_tsne[:len(Xs), 1], '.', label="source")
    plt.plot(X_tsne[len(Xs):, 0], X_tsne[len(Xs):, 1], '.', label="target")
    plt.legend(fontsize=14)
    plt.title("Encoded Space tSNE for the Source Only model")

    if SAVE_IMG_FLAG:
        plt.savefig("train_tsne_wo_da.png")
    else:
        plt.show()

    # save models
    _dir = os.path.join(args.base_model_root, "wo_da")
    if not os.path.exists(_dir):
        os.mkdir(_dir)
    finetunig.encoder.save(os.path.join(_dir, "resnet50_last_block.hdf5"))

def main_wo_da():

    print("Loading data")

    # data
    Xs, ys = get_Xy("amazon", path_to_folder=data_root)
    Xt, yt = get_Xy("webcam", path_to_folder=data_root)

    print("Loading the model")

    # base model
    resnet50 = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling="avg")

    first_layer = resnet50.get_layer('conv5_block2_out')
    #inputs = Input(first_layer.output_shape[1:])
    inputs = Input(first_layer.output.shape[1:])

    for layer in resnet50.layers[resnet50.layers.index(first_layer)+1:]:
        if layer.name == "conv5_block3_1_conv":
            x = layer(inputs)
        elif layer.name == "conv5_block3_add":
            x = layer([inputs, x])
        else:
            x = layer(x)

    first_blocks = Model(resnet50.input, first_layer.output)
    last_block = Model(inputs, x)

    last_block.summary()
    last_block.save(os.path.join(args.base_model_root, "resnet50_last_block.hdf5"))

    print("Launching training")

    train(Xs, ys, Xt, yt)

def main_w_da():
    pass

def main_eval():
    pass

if __name__ == "__main__":

    if EVALUATE:
        main_eval()
    elif USE_DA:
        main_w_da()
    else:
        main_wo_da()
