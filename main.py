from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import SGD, adam
import prepare_data as pd
import numpy
import math


def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def model():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='he_normal',
                     activation='relu', border_mode='valid', bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(nb_filter=64, nb_row=1, nb_col=1, init='he_normal',
                     activation='relu', border_mode='valid', bias=True))
    SRCNN.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='he_normal',
                     activation='linear', border_mode='valid', bias=True))
    # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    Adam = adam(lr=0.001)
    SRCNN.compile(optimizer=Adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return SRCNN


def train():
    srcnn_model = model()
    data, label = pd.read_training_data("./train.h5")
    # srcnn_model.load_weights("m_model_adam.h5")
    srcnn_model.fit(data, label, batch_size=128, nb_epoch=30)
    srcnn_model.save_weights("m_model_adam30.h5")


def predict():
    srcnn_model = model()
    srcnn_model.load_weights("m_model_adam30.h5")
    IMG_NAME = "butterfly_GT.bmp"
    INPUT_NAME = "input.jpg"
    OUTPUT_NAME = "pre_adam30.jpg"

    import cv2
    img = cv2.imread(IMG_NAME)
    shape = img.shape
    img = cv2.resize(img, (shape[1] / 2, shape[0] / 2), cv2.INTER_CUBIC)
    img = cv2.resize(img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    cv2.imwrite(INPUT_NAME, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0]
    pre = srcnn_model.predict(Y, batch_size=1)
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]

    print "bicubic:"
    print psnr(im1, im2)
    print "SRCNN:"
    print psnr(im1, im3)


if __name__ == "__main__":
    train()
    predict()
