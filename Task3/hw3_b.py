import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from cifar_extract import load_input, plot_img
#variables
image_height = 32
image_width = 32
image_depth = 3
train_data = []
label_names = None
dataset_dir = './data/cifar-10-batches-py'

#get data
train_data, label_names = load_input(dataset_dir)

mean_img = []
#mean img
for i in range(10):
    train_data[i] = train_data[i].reshape([-1,image_height*image_width*image_depth])
    avg = np.zeros_like(train_data[i][0])
    for j in range(len(train_data[i])):
        avg = avg+train_data[i][j]
    avg = avg/len(train_data[i])
    #avg = avg.reshape([-1,image_height,image_width,image_depth])
    avg = avg/255
    mean_img.append(avg)

dist_mat = []
#compute distances between mean images for each pair of classes
for i in range(10):
    for j in range(10):
        diff = mean_img[i] - mean_img[j]
        dist = np.dot(np.transpose(diff),diff)
        dist_mat.append(dist)
dist_mat= np.array(dist_mat)
dist_mat = dist_mat.reshape([10,10])

#Comnpute A
A = np.subtract(np.identity(10), 1./10. * np.ones(shape =(10,10)))
#compute W
W = -1./2.*np.dot(np.dot(A,dist_mat),np.transpose(A))

#get eigenvalue and eigenvectors
v, u = LA.eig(W)
v_sort = np.sort(v, axis = None)
v_sort = v_sort[::-1]

#get largest two eigenvalues
lambda_1 = v_sort[0]
lambda_2 = v_sort[1]
#get index of chosen eigenvalues
index_1 = v.tolist().index(lambda_1)
index_2 = v.tolist().index(lambda_2)
#get corresponding eigenvectors
u_1 = u[:,index_1]
u_2 = u[:,index_2]

#compute plots
U = np.array([u_1,u_2]).transpose()
lambda_r = np.diag(np.sqrt([lambda_1,lambda_2]))
X = np.dot(U,lambda_r).transpose()
#plot graph

fig, ax = plt.subplots()
ax.scatter(X[0],X[1])
for i,txt in enumerate(label_names):
    ax.annotate(txt, (X[0][i],X[1][i]))
plt.savefig('./output/problem_b.png')
