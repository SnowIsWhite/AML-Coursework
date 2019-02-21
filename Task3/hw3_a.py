import numpy as np
import os
from sklearn.decomposition import PCA
from cifar_extract import load_input, plot_img
import PIL
from PIL import Image
import matplotlib.pyplot as plt

dataset_dir = './data/cifar-10-batches-py'

#variables
train_data = [] #list of numpy lists
label_names = None
image_height = 32
image_width = 32
image_depth = 3

#extract cifar image from train batches files
train_data, label_names = load_input(dataset_dir)

mean_img = []
#mean image
for i in range(10):
    N = len(train_data[i])
    arr = np.zeros((image_height,image_width,image_depth),np.float32)
    for im in train_data[i]:
        imarray = np.array(Image.fromarray(im.astype('uint8')), dtype= np.float32)
        arr = arr+imarray/N
    arr = np.array(np.round(arr), dtype = np.uint8)

    mean_img.append(arr)
for i in range(10):
    plot_img(mean_img[i], '{}{}{}'.format("mean_",i,".png"))

#pca to 20 components
pca_list = []
error_list = []
for i in range(10):   
    pca = PCA(n_components=20)
    train_data[i] = train_data[i].reshape([-1,32*32*3])
    U= pca.fit_transform(train_data[i])
    var = (pca.explained_variance_)
    pca_list.append(var)
    filename = '{}{}{}'.format('./output/pca/category',i,".txt")
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as fp:
        for item in var:
            fp.write("%.8f "%item)
    
    #project to PCA
    U= U.transpose([1,0]) #20*5000
    mean_img[i] = mean_img[i].reshape([image_height*image_width*image_depth])
    proj = np.dot(U,train_data[i]-mean_img[i]) #20*3072
    proj = proj.reshape([-1,image_height,image_width,image_depth])
    for j in range(len(proj)):
        filename = '{}{}{}{}{}'.format('/pca/new_',i,'_',j,'th.png')
        plot_img(proj[j],filename)
    
    #back to original space
    U = U.transpose([1,0]) #5000*20
    ori = pca.inverse_transform(U) #5000*3072
    #plot one sample image
    ori = ori.reshape([-1,image_height,image_width,image_depth])
    ori = ori/255
    filename = '{}{}{}'.format('/pca/return_',i,'.png')
    plot_img(ori[0], filename)
    
    #error
    error = 0
    ori = ori*255
    ori = ori.reshape([-1,image_height*image_width*image_depth])
    for j in range(len(ori)):
        temp = train_data[i][j] - ori[j]
        temp = temp**2
        result = np.sum(temp)
        result = result/(image_height*image_width*image_depth)
        error = error+result
    error = error / len(ori)
    error_list.append(error)

#plot error list
x_axis = np.arange(10)
plt.bar(x_axis,error_list, align = 'center', alpha = 0.3)
plt.savefig('./output/pca/error_chart.png')
"""
new_img = []
#plot each pca components
for ind in range(10):
    X_i = train_data[ind][0]    #1*3072
    mean_X = np.mean(X_i)       #1
    X_mid = X_i - mean_X        #1*3072
    for i in range(len(X_i)):
        temp =0
        for j in range(20):
            U_j= pca_list[ind][j]
            temp = temp +  U_j*X_mid[i]*U_j
        X_i[i] = mean_X+temp
    new_img.append(X_i)
    X_i = X_i.reshape([-1,image_height,image_width,image_depth])
    print(X_i.shape)
    filename = '{}{}{}'.format('/pca/new_',ind,'.png')
    plot_img(X_i[0],filename)
"""
