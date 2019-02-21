from struct import unpack
import gzip
from numpy import zeros, uint8, float32, ravel

from pylab import imshow, show, cm

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

from argparse import ArgumentParser
import os.path
import cPickle as pickle

import matplotlib.pyplot as plt

def get_labeled_data(imagefile, labelfile, picklename):
    """
    Read input-vector (image) and target class (label, 0-9).

    Return
    ------
    dict
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        images = gzip.open(imagefile, 'rb')
        labels = gzip.open(labelfile, 'rb')

        # Read the binary data

        # We have to get big endian unsigned int. So we need '>I'

        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]

        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = labels.read(4)
        N = unpack('>I', N)[0]

        if number_of_images != N:
            raise Exception('The number of labels did not match '
                            'the number of images.')

        # Get the data
        x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
        y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            for row in range(rows):
                for col in range(cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    x[i][row][col] = (float(tmp_pixel) / 255)
            tmp_label = labels.read(1)
            y[i] = unpack('>B', tmp_label)[0]
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data


def view_image(image, label=""):
    imshow(image, cmap=cm.gray)
    show()

def save_image(image, name = ""):
    filename = './output/'+name
    plt.imsave(filename, image)


def classify(training, testing, HIDDEN_NEURONS, MOMENTUM, WEIGHTDECAY,
             LEARNING_RATE, LEARNING_RATE_DECAY, EPOCHS):
    INPUT_FEATURES = testing['rows'] * testing['cols']
    print("Input features: %i" % INPUT_FEATURES)
    CLASSES = 10
    trndata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)
    tstdata = ClassificationDataSet(INPUT_FEATURES, 1, nb_classes=CLASSES)

    for i in range(len(testing['x'])):
        tstdata.addSample(ravel(testing['x'][i]), [testing['y'][i]])
    for i in range(len(training['x'])):
        trndata.addSample(ravel(training['x'][i]), [training['y'][i]])

    # This is necessary, but I don't know why
    # See http://stackoverflow.com/q/8154674/562769
    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,
                       outclass=SoftmaxLayer)

    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,
                              verbose=True, weightdecay=WEIGHTDECAY,
                              learningrate=LEARNING_RATE,
                              lrdecay=LEARNING_RATE_DECAY)
    print("Start training")
    for i in range(EPOCHS):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                 trndata['class'])
        tstresult = percentError(trainer.testOnClassData(
                                 dataset=tstdata), tstdata['class'])

        print("epoch: %4d" % trainer.totalepochs,
              "  train error: %5.2f%%" % trnresult,
              "  test error: %5.2f%%" % tstresult)
    return fnn
