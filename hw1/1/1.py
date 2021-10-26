import numpy as np
import math
import os
import keras.utils.np_utils as utils
import matplotlib.pyplot as plt

# parameter to DNN
hidden_layer_num = 2
hidden1_num = 100
hidden2_num = 50 # 50, 2 for latent
output_num = 10
learning_rate = 0.1
beta = 0.8
iteration = 60
mini_batch = 200
init_weight = {'zero':True,'random':False}
weight_state = True
epoch_num = 0
epoch_iter = 500  # 500
# momentum
v_w1 = np.zeros((28*28,hidden1_num))
v_b1 = np.zeros((hidden1_num))
v_w2 = np.zeros((hidden1_num,hidden2_num))
v_b2 = np.zeros((hidden2_num))
v_w3 = np.zeros((hidden2_num,output_num))
v_b3 = np.zeros((output_num))

# plot curve
train_accu_list = []
ave_loss_list = []
test_accu_list = []
epoch_list = []

def read_data():
    test_path = os.path.join('problem1-DNN','test.npz')
    train_path = os.path.join('problem1-DNN','train.npz')
    
    test_data = np.load(test_path)
    test_img = test_data['image'].reshape(5768,784)             # size:5768x784 (image size:28x28=784)
    test_label = utils.to_categorical(test_data['label'], 10)   # size:5678x10  (label:0~9)

    train_data = np.load(train_path)
    train_img = train_data['image'].reshape(12000,784)          # size:12000x784
    train_label = utils.to_categorical(train_data['label'],10)  # size:12000x10

    # # to check the train data index
    # index_list = {}
    # for i in range(10):
    #     index_list[i] = np.where(train_data['label'] == i)
    # for label,idxs in index_list.items():
    #     print('category:',label,'num:',len(idxs[0]),'\nitems:',idxs)

    # # show img
    # plt.imshow(train_data['image'][0],cmap='gray')
    # plt.title('Class '+str(train_data['label'][0]))
    # plt.show()
    
    return train_img,train_label,test_img,test_label

def ReLU(z):
    return np.maximum(z,0)
    # if isinstance(z,np.ndarray):
    #     z_re = z.copy()
    #     # print('ReLU z size:',len(z_re))
    #     for i in range(len(z_re)):
    #         if z_re[i] <= 0:
    #             z_re[i] = 0
    #     return z_re
    # else:
    #     if z > 0:
    #         return z
    #     else:
    #         return 0.001*z

def ReLU_diff(z):
    if isinstance(z,np.ndarray):
        z_re = z.copy()
        for i in range(len(z_re)):
            if z_re[i] <= 0:
                z_re[i] = 0
            else:
                z_re[i] = 1
        return z_re
    else:
        if z > 0:
            return 1
        else:
            return 0.000001*z

def sigmoid(z):
    return 1 / (1+np.exp(-np.clip(z,-500,500)))

def sigmoid_diff(x):
    return np.multiply(sigmoid(x),(1-sigmoid(x)))

def softmax(a):
    y = a.copy()
    # print(y)
    min_a = np.max(a)
    # print(min_a)
    # for i in range(len(a)):
    #     y[i] = float(np.exp(float(a[i])-min_a))
        # print(y[i])
    y = np.exp(y-min_a)
    y /= sum(y)
    return y

def cross_entropy(y,t,confu_matrix):  # t:train label target, y:approximated y
    index = list(t.copy()).index(1)
    temp = y[index]
    if temp<0.000001:
        temp = 0.0000001
    i = list(y.copy()).index(max(y))
    # print(i,index)
    confu_matrix[i][index] += int(1)
    return -1*math.log(temp),confu_matrix

def build_hidden(zero_initial:bool,input_size,output_size): # zero_initial = True: using zero initializations for model weights matrix: input x output
    if zero_initial:
        tensor = np.zeros((input_size,output_size))
        bias = np.full((output_size),0.01)
    else:
        tensor = np.random.normal(size=(input_size,output_size))
        bias = np.random.rand(output_size)*0.01
    return tensor,bias


def shuffle_train(train_x,train_t):
    a = train_x.copy()
    b = train_t.copy()

    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size//len(a)].reshape(a.shape)
    b2 = c[:, a.size//len(a):].reshape(b.shape)

    # show img
    # plt.imshow(a2[0].reshape(28,28),cmap='gray')
    # plt.title('Class '+str(b2[0]))
    # plt.show()
    return a2,b2

def normalization(x):
    # mean = np.sum(x)/(x.size)
    # A = x-mean
    # var = np.sum(A*A)/x.size
    # return (x-mean)/(var**0.5)
    return x.copy()/255

def back_prop(h1_w,h2_w,h3_w,h1_b,h2_b,h3_b,x,y,t,a1,a2,a3,z1,z2,z3):
    global v_b1,v_b2,v_b3,v_w1,v_w2,v_w3
    dy = np.subtract(y , t)
    # print(dy)
    db3 = np.multiply(dy,sigmoid_diff(z3))
    v_b3 = beta*v_b3 + learning_rate*db3
    # print(v_b3)
    b3_n = np.subtract(h3_b,v_b3)
    dw3 = np.dot(np.reshape(a2.copy(),(hidden2_num,1)),np.reshape(v_b3,(1,output_num)))
    v_w3 = beta*v_w3 + learning_rate*dw3
    # print(v_w3)
    w3_n = np.subtract(h3_w,v_w3)
    db2 = np.multiply(sigmoid_diff(z2),np.dot(db3,w3_n.T))
    v_b2 = beta*v_b2 + learning_rate*db2
    b2_n = np.subtract(h2_b , v_b2)
    dw2 = np.dot(np.reshape(a1.copy(),(hidden1_num,1)),np.reshape(v_b2,(1,hidden2_num)))
    v_w2 = beta*v_w2 + learning_rate*dw2
    w2_n = np.subtract(h2_w , v_w2)
    db1 = np.multiply(ReLU_diff(z1),np.dot(db2,w2_n.T))
    v_b1 = beta*v_b1 + learning_rate*db1
    b1_n = np.subtract(h1_b , v_b1)
    dw1 = np.dot(np.reshape(x.copy(),(28*28,1)),np.reshape(v_b1,(1,hidden1_num)))
    v_w1 = beta*v_w1 + learning_rate*dw1
    w1_n = np.subtract(h1_w , v_w1)
    # print('original w3=',h3_w)
    # print('BP dw3=',dw3)
    return  w1_n,w2_n,w3_n,b1_n,b2_n,b3_n

def forward_prop(train_x,train_y,h1_w,h2_w,h3_w,h1_b,h2_b,h3_b):
    confu_matrix = np.zeros((10,10))
    ave_loss = 0
    for it in range(iteration):
        w1,b1 = build_hidden(init_weight['zero'],28*28,hidden1_num)
        w2,b2 = build_hidden(init_weight['zero'],hidden1_num,hidden2_num)
        w3,b3 = build_hidden(init_weight['zero'],hidden2_num,output_num)
        train_data_num = 0

        for batch in range(mini_batch):
            i = it*mini_batch + batch
            x = normalization(train_x[i])
            t = train_y[i]
            # print('The',i,'-th train data')
            # print(x)
            # print(t)
            
            try:
                z1 = np.dot(x,h1_w) + h1_b          # x:1x784, h1_w:784xhidden1_num(100)
                a1 = ReLU(z1)                       # a1:1xhidden1_num(100)->first layer output
                z2 = np.dot(a1,h2_w) + h2_b         # z2:1xhidden2_num(50)
                a2 = sigmoid(z2)                    # a2:1xhidden2_num(50)->second layer output
                z3 = np.dot(a2,h3_w) + h3_b         # z3:1xoutput_num(10)
                a3 = sigmoid(z3)                    # a3:1xoutput_num(10)->third layer output
                y = softmax(a3)                     # y:1xoutput_num(10)->approxiated output y(hat)
            except:
                print('h1_w=',h1_w)
                print('h2_w=',h2_w)
                print('h3_w=',h3_w)
                print('a1=',a1)
                print('a2=',a2)
                print('a3=',a3)
            
            # print('y = ',y)
            # print('t = ',t)
            # print('h1_w=',h1_w)
            # print('h2_w=',h2_w)
            # print('h3_w=',h3_w)
            # print('a1=',a1)
            # print('a2=',a2)
            # print('a3=',a3)
            train_loss,confu_matrix = cross_entropy(y,t,confu_matrix)
            # print('train loss = ',train_loss)
            ave_loss += train_loss
            w1_n,w2_n,w3_n,b1_n,b2_n,b3_n = back_prop(h1_w,h2_w,h3_w,h1_b,h2_b,h3_b,x,y,t,a1,a2,a3,z1,z2,z3)
            w1 += w1_n
            w2 += w2_n
            w3 += w3_n
            b1 += b1_n
            b2 += b2_n
            b3 += b3_n
            train_data_num += 1
        h1_w = w1/train_data_num
        h2_w = w2/train_data_num
        h3_w = w3/train_data_num
        h1_b = b1/train_data_num
        h2_b = b2/train_data_num
        h3_b = b3/train_data_num
    
    return confu_matrix,ave_loss,h1_w,h2_w,h3_w,h1_b,h2_b,h3_b

def plot_latent_features(w1,w2,y):
    if hidden2_num!=2:
        return
    
    if weight_state:
        figname = '2D feature '+str(epoch_num)+' epoch(zero weights)'
    else:
        figname = '2D feature '+str(epoch_num)+' epoch(random weights)'
    label = ['0','1','2','3','4','5','6','7','8','9']
    plt.figure(figname,figsize=(12,8))
    plt.suptitle(figname,fontsize=18)
    plt.xlabel('a1',fontsize=14)
    plt.ylabel('a2',fontsize=14)
    scatter = plt.scatter(w1,w2,c=y)
    plt.legend(*scatter.legend_elements(),loc='upper right',title='classes')
    save_fig_path = os.path.join('DNN_output',figname)
    plt.savefig(save_fig_path)
    return

def test_valid(test_x,test_y,h1_w,h2_w,h3_w,h1_b,h2_b,h3_b):
    test_matrix = np.zeros((10,10))
    latent_w1 = []
    latent_w2 = []
    latent_y = []
    for i in range(5768):
        x = normalization(test_x[i])
        t = test_y[i]
        try:
            z1 = np.dot(x,h1_w) + h1_b      # x:1x784, h1_w:784xhidden1_num(100)
            a1 = ReLU(z1)                   # a1:1xhidden1_num(100)->first layer output
            z2 = np.dot(a1,h2_w) + h2_b     # z2:1xhidden2_num(50)
            a2 = sigmoid(z2)                # a2:1xhidden2_num(50)->second layer output
            z3 = np.dot(a2,h3_w) + h3_b     # z3:1xoutput_num(10)
            a3 = sigmoid(z3)                # a3:1xoutput_num(10)->third layer output
            y = softmax(a3)                 # y:1xoutput_num(10)->approxiated output y(hat)
            index = list(t.copy()).index(1)
            i = list(y.copy()).index(max(y))
            test_matrix[i][index] += 1
            if (epoch_num == 20) or (epoch_num == 80):
                latent_w1.append(a2[0])
                latent_w2.append(a2[1])
                latent_y.append(i)
        except:
            print('test error')
    if (epoch_num == 20) or (epoch_num == 80):
        plot_latent_features(latent_w1,latent_w2,latent_y)
    return  test_matrix

if __name__ == "__main__":
    train_img,train_label,test_img,test_label = read_data()
    if weight_state:
        # initial the zero weights
        fig_name = '_zero_weight'
        weight_state = True
        h1_w,h1_b = build_hidden(init_weight['zero'],28*28,hidden1_num)
        h2_w,h2_b = build_hidden(init_weight['zero'],hidden1_num,hidden2_num)
        h3_w,h3_b = build_hidden(init_weight['zero'],hidden2_num,output_num)
    else:
        # initial the random weights
        fig_name = '_random_weight'
        weight_state = False
        h1_w,h1_b = build_hidden(init_weight['random'],28*28,hidden1_num)
        h2_w,h2_b = build_hidden(init_weight['random'],hidden1_num,hidden2_num)
        h3_w,h3_b = build_hidden(init_weight['random'],hidden2_num,output_num)
    
    c_matrix = np.zeros((10,10))
    t_matrix = np.zeros((10,10))
    fig,(ax1, ax2, ax3) = plt.subplots(1,3,figsize=(19,8))
    plt.ion()
    for i in range(epoch_iter):
        train_x,train_t = shuffle_train(train_img,train_label)
        # print('h1_w:',h1_w[0][0])
        confu_matrix,ave_loss,h1_w,h2_w,h3_w,h1_b,h2_b,h3_b = forward_prop(train_x,train_t,h1_w,h2_w,h3_w,h1_b,h2_b,h3_b)
        epoch_num += 1
        c_matrix = confu_matrix
        epoch_list.append(epoch_num)
        print('===================',epoch_num,'epoch ===================')
        # print('h1_w:',h1_w[0][0])
        print('confusion matrix:\n',confu_matrix)
        train_accu = np.trace(confu_matrix)/np.sum(confu_matrix)
        train_accu_list.append(train_accu)
        print('train accuracy:',train_accu,'(training data num =',np.sum(confu_matrix),')')
        test_matrix = test_valid(test_img,test_label,h1_w,h2_w,h3_w,h1_b,h2_b,h3_b)
        t_matrix = test_matrix
        test_accu = np.trace(test_matrix)/np.sum(test_matrix)
        test_accu_list.append(test_accu)
        print('test accuracy:',test_accu,'(test data num =',np.sum(test_matrix),')')
        ave_loss /= np.sum(confu_matrix)
        ave_loss_list.append(ave_loss)
        print('train loss:',ave_loss)

        # plt.subplot(3,1,1)
        ax1.cla()
        ax1.plot( epoch_list,train_accu_list,'-b')
        ax1.set_title('train accuracy ('+str(round(train_accu*100,2))+'%)',fontsize=18)
        ax1.set_xlabel('epoch number',fontsize=14)
        ax1.set_ylabel('accuracy rate(%)',fontsize=14)
        # plt.subplot(3,1,2)
        ax2.cla()
        ax2.plot( epoch_list,test_accu_list,'-b')
        ax2.set_title('test accuracy ('+str(round(test_accu*100,2))+'%)',fontsize=18)
        ax2.set_xlabel('epoch number',fontsize=14)
        ax2.set_ylabel('accuracy rate(%)',fontsize=14)
        # plt.subplot(3,1,3)
        ax3.cla()
        ax3.plot( epoch_list,ave_loss_list,'-b')
        ax3.set_title('train loss ('+str(round(ave_loss,2))+')',fontsize=18)
        ax3.set_xlabel('epoch number',fontsize=14)
        ax3.set_ylabel('cross entropy',fontsize=14)
        plt.pause(0.1)
    plt.ioff()
    save_path = os.path.join('DNN_output',str(epoch_iter)+fig_name)
    with open(os.path.join('DNN_output',str(epoch_iter)+fig_name+'_confusion_matrix.txt'),'w') as f:
        mat = np.matrix(c_matrix)
        np.savetxt(f,mat,fmt='%d',header='train')
        mat2 = np.matrix(t_matrix)
        np.savetxt(f,mat2,fmt='%d',header='test')
        np.savetxt(f,np.matrix(h1_w),fmt='%.3f',header='h1 w')
        np.savetxt(f,np.matrix(h2_w),fmt='%.3f',header='h2 w')
        np.savetxt(f,np.matrix(h3_w),fmt='%.3f',header='h3 w')
        np.savetxt(f,np.matrix(h1_b),fmt='%.3f',header='h1 b')
        np.savetxt(f,np.matrix(h2_b),fmt='%.3f',header='h2 b')
        np.savetxt(f,np.matrix(h3_b),fmt='%.3f',header='h3 b')
    fig.savefig(save_path)
    print('=========== end train ===========')
    plt.show()
    
    
    