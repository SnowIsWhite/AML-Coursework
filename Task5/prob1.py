import numpy as np
from sklearn.cluster import KMeans

vocab_list_dir = './data/vocab_list.txt'
vocab_cnt_dir = './data/docword.nips.txt'

def read_vocab_list(data_dir):
    voc_list = list(open(data_dir).read().split())
    return voc_list

def read_vocab_cnt(data_dir):
    with open(data_dir) as f:
        word_dict ={}
        D = f.readline().rstrip('\n')
        W = f.readline().rstrip('\n')
        N = f.readline().rstrip('\n')
        for line in f:
            (key1, key2, val) = line.rstrip('\n').split()
            if int(key1) not in word_dict:
                word_dict[int(key1)] = {}
            word_dict[int(key1)][int(key2)] = int(val)
    return int(D), int(W), int(N), word_dict
"""
def calculate_wij(X, Pj, pi):
    #numerator
    val = X[:,:,None]*np.log(Pj[None,:,:]) #D V T
    val = val.transpose([1,0,2]) #  V D T
    val = np.reshape(val, [V, -1])
    val = np.sum(val, axis = 0) # D*T,
    val = np.reshape(val, [D,-1])
    numerator = (val + np.log(pi[None,:]))
    
    #denumerator
    denumerator = np.sum(np.exp(numerator.transpose()),axis=0) #D,
    wij = np.exp(numerator - (denumerator[:,None]*1.0))
    return wij, numerator
"""
def EM(num_cluster):
    #data preprocess
    voc_list = read_vocab_list(vocab_list_dir)
    D,W,N,word_dict = read_vocab_cnt(vocab_cnt_dir)

    X = np.zeros(shape = [D,W]) #doc, voc
    for key1 in word_dict:
        for key2 in word_dict[key1]:
            X[key1-1][key2-1] = word_dict[key1][key2]
    X= X.transpose()
    #remove non occuring words
    X = X[~np.all(X==0, axis=1)] #12375,1500
    V = X.shape[0]
    X= X.transpose() #1500, 12375

    #Kmeans
    kmeans = KMeans(n_clusters = num_cluster).fit(X)
    num_topic = kmeans.cluster_centers_.shape[0]
    c_result = kmeans.labels_
    
    Pj = np.ones(shape= [num_topic,V]) #similar to add one smoothing
    for i in range(num_topic):
        for j in range(D):
            if c_result[j] == i:
                Pj[i] += X[j]
    Pj = Pj/ (np.sum(Pj, axis = 1)*1.0)[:,None]
    Pj = Pj.transpose()
    
    unique, counts = np.unique(kmeans.labels_, return_counts = True)
    d = dict(zip(unique, counts))
    pi = list(d.values()/(sum(d.values())*1.))
    pi = np.array(pi)
    
    diff, diff_p = 100,0
    
    #numerator
    val = X[:,:,None]*np.log(Pj[None,:,:]) #D V T
    val = val.transpose([1,0,2]) #  V D T
    val = np.reshape(val, [V, -1])
    val = np.sum(val, axis = 0) # D*T,
    val = np.reshape(val, [D,-1])
    numerator = (val + np.log(pi[None,:]))
    
    #denumerator
    denumerator = np.sum(np.exp(numerator.transpose()),axis=0) #D,
    wij = np.exp(numerator - (denumerator[:,None]*1.0))# D T
    
    Q = np.sum(numerator*wij)
    print(Q)

    while abs(diff - diff_p) > 0.1:
        #M step
        dot = np.dot(X.transpose(), wij)
        print(dot)
        Pj = dot/np.sum(dot, axis = 0)[None,:] #V,T
        print(Pj)
    
        pi = np.sum(wij, axis = 0)/D
        
        #Estep
         #numerator
        val = X[:,:,None]*np.log(Pj[None,:,:]) #D V T
        val = val.transpose([1,0,2]) #  V D T
        val = np.reshape(val, [V, -1])
        val = np.sum(val, axis = 0) # D*T,
        val = np.reshape(val, [D,-1])
        numerator = (val + np.log(pi[None,:]))
    
        #denumerator
        denumerator = np.sum(np.exp(numerator.transpose()),axis=0) #D,
        wij = np.exp(numerator - (denumerator[:,None]*1.0))# D T
    
        Q_p = np.sum(numerator*wij)
        diff= diff_p
        diff_p = abs(Q-Q_p)

        print(diff_p)
        Q = Q_p


EM(3)
#init p, pi

#calculate wij
#EM


