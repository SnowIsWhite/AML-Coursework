from get_mnist import*
import numpy as np

width = 28
height = 28
noise = 0.02
batch_size = 500
testing = get_labeled_data('./data/train-images-idx3-ubyte.gz', './data/train-labels-idx1-ubyte.gz','testing')
data = testing['x'][0:1*batch_size] #batch, 28,28

def neighboring_idx(idx):
    neighbors = []
    row = (int)(idx/28)
    col = (int)(idx%28)

    if row -1 >0:
        neighbors.append((row-1)*28+col)
    if row +1 <28:
        neighbors.append((row+1)*28+col)
    if col -1 >0:
        neighbors.append(row*28+col-1)
    if col +1 <28:
        neighbors.append(row*28+col+1)

    return neighbors

#1d array
data = np.array(data)
data = np.reshape(data, [-1,width*height]) #500,28*28
data[data>=0.5] = 1.
data[data<0.5] = -1.

#noise
X = np.copy(data)
for i in range(batch_size):
    idx = np.random.choice(height*width, (int)(height*width*noise), replace = False)
    X[i][idx] = -data[i][idx]

#mean field inference
def init_pi(init_pi):
    if init_pi == 1:
        # 0.5 for all
        pi = np.ones_like(X)/2.
    elif init_pi == 2:
        #1 or 0 depending on X_i
        pi = np.ones_like(X)
        pi[X==-1.] = 0.
    elif init_pi == 3:
        #random
        pi = np.random.rand(batch_size,height*width)
    return pi

def boltzman(pi, h_theta, x_theta=2.,name = 'sample'):
    diff = 0.5
    while diff >1e-8:
        #calculate pi
        pi_temp = np.copy(pi)
        for i in range(height*width):
            core = (np.sum((h_theta*(2*pi[:,neighboring_idx(i)]-1)+(x_theta*X[:,i])[:,None]),axis=1))
            numer = np.exp(core)
            denum = np.exp(-1*core) + np.exp(core)
            pi[:,i] = numer/denum
        diff = abs(np.mean(np.subtract(pi,pi_temp)))
        print(diff)
    pi[pi>=0.5] = 1.
    pi[pi<0.5] = -1.
    return pi

def prob1(data,X,pi_param):
    pi = init_pi(pi_param)
    pi = boltzman(pi,h_theta=0.2,name = 'prob1')
    diff = np.sum(abs(data- pi), axis = 1)
    frac = 1.-(np.sum(diff/2.)/(batch_size*height*width))
    print( '{}{}{}{}'.format('correct fraction: ',pi_param,' and ', frac))
    
    worst = np.argmax(diff)
    worst_frac = 1.-(np.max(diff)/2.)/(height*width)
    best = np.argmin(diff)
    best_frac = 1.-(np.min(diff)/2.)/(height*width)
    print('{}{}{}{}'.format('worst_frac: ',pi_param,' and ', worst_frac))
    print('{}{}{}{}'.format('best_frac: ',pi_param,' and ', best_frac))
    data = np.reshape(data, [batch_size,height,width])
    X = np.reshape(X, [batch_size,height,width])
    pi = np.reshape(pi, [batch_size,height,width])

    if pi_param ==1:
        save_image(data[best],'prob1_best_origin_1')
        save_image(X[best], 'prob1_best_noise_1')
        save_image(pi[best], 'prob1_best_rec_1')
        save_image(data[worst],'prob1_worst_orgin_1')
        save_image(X[worst], 'prob1_worst_noise_1')
        save_image(pi[worst],'prob1_worst_rec_1')
    elif pi_param ==2:
        save_image(data[best],'prob1_best_origin_2')
        save_image(X[best], 'prob1_best_noise_2')
        save_image(pi[best], 'prob1_best_rec_2')
        save_image(data[worst],'prob1_worst_orgin_2')
        save_image(X[worst], 'prob1_worst_noise_2')
        save_image(pi[worst],'prob1_worst_rec_2')
    elif pi_param ==3:
        save_image(data[best],'prob1_best_origin_3')
        save_image(X[best], 'prob1_best_noise_3')
        save_image(pi[best], 'prob1_best_rec_3')
        save_image(data[worst],'prob1_worst_orgin_3')
        save_image(X[worst], 'prob1_worst_noise_3')
        save_image(pi[worst],'prob1_worst_rec_3')

def prob2(data,X,pi_param):
    h_theta = [-1,-0.5,0,0.5,0.8,1,5,8,10,15,20]
    tpr_arr = []
    fpr_arr = []
    data = np.reshape(data, [batch_size*height*width])
    for cnt, val in enumerate(h_theta):
        pi = init_pi(pi_param)
        pi = boltzman(pi, h_theta = val)
        
        pi = np.reshape(pi, [batch_size*height*width])
        pi_t = 2*pi
        eval_t = data+pi_t

        unique, counts = np.unique(eval_t, return_counts = True)
        d = dict(zip(unique,counts))
        TP = d[3]
        FN = d[-1]
        FP = d[1]
        TN = d[-3]

        TPR = (TP/(float)(TP+FN))
        FPR = (FP/(float)(FP+TN))
        tpr_arr.append(TPR)
        fpr_arr.append(FPR)
        
        pi = np.reshape(pi, [batch_size, height*width])
    with open('./output/roc2.txt', 'w') as f:
        for cnt, val in enumerate(h_theta):
            f.write('{}{}{}{}'.format(tpr_arr[cnt],' ',fpr_arr[cnt],'\n'))
    f.close()

#prob1(data,X,1)
#prob1(data,X,2)
#prob1(data,X,3)
prob2(data,X,3)
