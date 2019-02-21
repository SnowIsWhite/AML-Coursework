import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cifar_extract import load_input, plot_img
from numpy import linalg as LA

dataset_dir = './data/cifar-10-batches-py'
#variables
train_data= []  #list of numpy lists 10 indicies
mean_data = []
dist_mat = []   #distance matrix with new definition of similiarity
reconstruct_img =[]
label_names = None
image_height = 32
image_width = 32
image_depth =3
n_classes = 10
#get data
train_data, label_names = load_input(dataset_dir)

for i in range(n_classes):
    train_data[i] = train_data[i]/255

#get mean of all classes
for i in range(n_classes):
    train_data[i] = train_data[i].reshape([-1,image_height*image_width*image_depth])
    avg = np.zeros_like(train_data[i][0])
    for j in range(len(train_data[i])):
        avg = avg+train_data[i][j]
    avg = avg/len(train_data[i])
    avg = avg/255
    put = avg.reshape([1,image_height*image_width*image_depth])
    mean_data.append(put) #1,3072

eig_vec_list = []
reduced_list = []
#get deduced value and eigenvectors for each classes
for i in range(n_classes):
    pca = PCA(n_components=20)
    reduced = pca.fit_transform(train_data[i])
    #get eigenvectors
    eig_vec = pca.components_ #20,3072
    eig_vec_list.append(eig_vec)
    reduced_list.append(reduced)#5000,20

#construct distance mat
for i in range(n_classes):
    for j in range(n_classes):
        dist_mat.append([0.])

dist_mat = np.array(dist_mat)
dist_mat = dist_mat.reshape([10,10])

#construct E function
def __get_E(A,B):
    #duplicate eigenvectors
    """
    #temp = np.zeros_like([1,image_height*image_width*image_depth])#1,3072
    #temp = np.dot(np.dot(U_B[j].transpose(),(train_data[A]-mean_data[A])),U_B[j].transpose()).transpose()
    """
    #reconstruct image
    temp = np.dot(train_data[A]-mean_data[A],np.dot(eig_vec_list[B].transpose(),eig_vec_list[B]))
    new_img = mean_data[A]+temp

    #calculate distance
    diff = (train_data[A] - new_img)
    dist = np.dot(diff,np.transpose(diff))
    result = np.sum(dist,axis = 0)
    result = np.sum(result)/((float)(len(train_data[A])))
    return result

#put results in matrix
for i in range(n_classes):
    for j in range(n_classes):
        if i>j:
            dist_mat[i][j] = dist_mat[j][i]
        elif i==j:
            dist_mat[i][j] = __get_E(i,j)
        else:
            dist_mat[i][j] = 1./2.*(__get_E(i,j) + __get_E(j,i))

#Principal Coordinate Analysis
#Compute A
A = np.subtract(np.identity(n_classes), 1./10.*np.ones(shape =(n_classes,n_classes)))
#compute W
W = -1./2.*np.dot(np.dot(A,dist_mat),np.transpose(A))

#get eigenvalue and eigenvecors
v, u = LA.eig(W)
v_sort = np.sort(v,axis=None)
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

#plot
u_vec = np.array([u_1,u_2]).transpose()
lambda_r = np.diag(np.sqrt([lambda_1,lambda_2]))
X = np.dot(u_vec,lambda_r).transpose()

fig, ax = plt.subplots()
ax.scatter(X[0],X[1])
for i,txt in enumerate(label_names):
    ax.annotate(txt, (X[0][i],X[1][i]))
plt.savefig('./output/problem_c.png')
