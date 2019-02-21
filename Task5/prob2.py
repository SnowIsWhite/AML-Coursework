import numpy as np
from scipy import ndimage as img
from scipy.misc import imsave
from sklearn.cluster import KMeans
from numpy.core.umath_tests import matrix_multiply

def calculate_wij(X,mu,pi):
    mu_p = np.reshape(mu, [-1,1,3])
    sub = X[:, None,:,:] - mu_p[None,:,:,:]
    temp = np.reshape(sub, [sub.shape[0]*sub.shape[1],1,-1])
    cov = np.exp(-(1/2.)*matrix_multiply(temp, temp.transpose([0,2,1])))
    cov = np.reshape(cov, [sub.shape[0], sub.shape[1]])
       
    numerator = cov * pi
    denumerator = (np.sum(cov * pi, axis = 1))
    wij = numerator/denumerator[:,None]
    return wij, cov

def EM(image_dir, num_cluster):
    #read image
    data = img.imread(image_dir)
    height = data.shape[0]
    width = data.shape[1]
    depth = data.shape[2]
    
    #data preprocess
    data = data/255.*10.
    data = data.transpose([2,0,1])
    data = np.reshape(data, [depth,height*width])
    mean = np.mean(data, axis = 1)
    data = data.transpose() - mean
    data = np.reshape(data.transpose(), [depth, height, width])
    data.transpose([1,2,0])
    
    #K-means
    data = np.reshape(data, [height*width, depth])
    kmeans = KMeans(n_clusters = num_cluster).fit(data)

    # initial mu and pi
    mu = kmeans.cluster_centers_ # (num_cluster, 3)
    unique, counts = np.unique(kmeans.labels_, return_counts = True)
    d = dict(zip(unique, counts))
    pi = list(d.values())/(sum(d.values())*1.)
    pi = np.array(pi)

    X = np.reshape(data, [height*width, 1, depth])
    diff = 100
    diff_p = 0
    #E step
    wij, cov = calculate_wij(X,mu,pi)
    Q = np.sum((cov + np.log(pi))*wij)
    while abs(diff_p-diff)>0.1:
        #M step
        m_n = np.sum(X * wij[:,:,None], axis = 0)
        m_d = np.sum(wij, axis= 0)
        mu = m_n/m_d[:,None]
        
        pi = np.sum(wij, axis = 0)/height*width
        
        #E step
        wij, cov = calculate_wij(X,mu,pi)
        Q_p = np.sum((cov+np.log(pi))*wij)
        diff = diff_p
        diff_p = np.absolute(Q - Q_p)
        
        print(diff)
        Q = Q_p

    #change image
    idx = np.argmax(wij, axis =1) #height*widht,
    
    mu = np.expand_dims(mu, axis = 0)
    mu = np.repeat(mu, height*width, axis =0)
    """  
    result = np.ones([height,width, depth])
    for i in range(height):
        for j in range(width):
            result[i][j] = mu[idx[i*j]]
    result = np.reshape(result, [height*width, depth])
    """

    result = mu[np.array(range(height*width)),idx] 
    #transform into rgb
    result = result + mean[None,:]
    result = np.reshape(result, [height, width, depth])
    result *= 255/10.
    return result

pic1 = './data/prob2_1.jpg'
pic2 = './data/prob2_2.jpg'
pic3 = './data/prob2_3.jpg'

pic31 = EM(pic3, 20)
imsave('./output/pic31.jpg',pic31)
pic32 = EM(pic3, 20)
imsave('./output/pic32.jpg',pic32)
pic33 = EM(pic3, 20)
imsave('./output/pic33.jpg',pic33)
pic34 = EM(pic3, 20)
imsave('./output/pic34.jpg',pic34)
pic35 = EM(pic3, 20)
imsave('./output/pic35.jpg',pic35)

pic1_10= EM(pic1, 10)
imsave('./output/pic1_10.jpg',pic1_10)

pic1_20= EM(pic1, 20)
imsave('./output/pic1_20.jpg',pic1_20)

pic1_50= EM(pic1, 50)
imsave('./output/pic1_50.jpg',pic1_50)

pic2_10= EM(pic2, 10)
imsave('./output/pic2_10.jpg',pic2_10)

pic2_20= EM(pic2, 20)
imsave('./output/pic2_20.jpg',pic2_20)

pic2_50= EM(pic2, 50)
imsave('./output/pic2_50.jpg',pic2_50)

pic3_10= EM(pic3, 10)
imsave('./output/pic3_10.jpg',pic3_10)

pic3_20= EM(pic3, 20)
imsave('./output/pic3_20.jpg',pic3_20)

pic3_50 = EM(pic3, 50)
imsave('./output/pic3_50.jpb', pic3_50)
