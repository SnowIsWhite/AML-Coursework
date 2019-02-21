import numpy as np
import cPickle
import re
import os
import matplotlib.pyplot as plt

dataset_dir = './data/cifar-10-batches-py'
r_data_file = re.compile('^data_batch_\d+')

#variables
image_height = 32
image_width = 32
image_depth = 3

#plot image
def plot_img(img_data, name):
    #img_data = img_data/255
    fig = plt.figure(figsize = (4,4))
    plt.axis = ("off")
    plt.imshow(img_data)
    plt.savefig('./output/'+name)
    plt.close(fig)

#extract data object from data stream
def unpickle(relpath):
    with open(relpath, 'rb') as fp:
        d = cPickle.load(fp)
    return d

#preprocess input
def preprocess_input(data = None):
    global image_height, image_width, image_depth
    #data = data/255
    data = data.reshape([-1,image_depth,image_height,image_width])
    data = data.transpose([0,2,3,1])
    data = data.astype(np.float32)
    return data

def load_input(dataset_dir):
    #divide trainig data into categories
    train_data = [[] for _ in range(10)]
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            m = r_data_file.match(f)
            if m:
                relpath = os.path.join(root,f)
                d = unpickle(os.path.join(root,f))
                temp_labels = d['labels']
                temp_data = d['data']
                for i,_ in enumerate(temp_labels):
                    index = temp_labels[i]
                    train_data[index].append(temp_data[i])
    #print(len(train_data[0]))
    #print(len(train_data[4]))
    for i in range(10):
        train_data[i] = (np.concatenate(train_data[i]).astype(np.float32))
        train_data[i] = preprocess_input(data= train_data[i])

    label_names = unpickle(os.path.join(dataset_dir, 'batches.meta'))['label_names']
    print("CIFAR 10 data extraction completed")
    return train_data, label_names

#load_input('./data/cifar-10-batches-py')


