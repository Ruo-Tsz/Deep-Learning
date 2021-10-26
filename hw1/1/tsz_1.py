import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os
import keras.utils.np_utils as utils
from tqdm import tqdm, trange
from datetime import datetime

# remove sys path of ros cv2 to enable import cv2 in conda lib
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

project_path = os.path.dirname(os.path.realpath(__file__))
# project_path = "/home/ee904/tsz/Deep Learning/hw1/1"
layer_num = 3
# epoch_num = 500
batch_size = 100 #size of mini-batch
learning_rate = 0.001
# learning_rate = 0.01
fig_output_dir = os.path.join(project_path,'fig','4_layer_leaky',str(batch_size))

# ReLU would failed to update gradient (zero output and zero gradient, dead ReLU) in both zero_initial and latent nodes case
# try leaky ReLU 
leaky_alpha = 0.5

if not os.path.exists(fig_output_dir):
    os.makedirs(fig_output_dir)

def load_data(data_type="train"):
    data_path = ""
    data_size = 0
    if data_type == "train":
        data_path = os.path.join(project_path, "data", "train.npz")
        data_size = 12000
    elif data_type == "test":
        data_path = os.path.join(project_path, "data", "test.npz")
        data_size = 5768
    else:
        print("Please provide the valid dataset type (train/test)")
        exit(-1)

    raw_data = np.load(data_path)

    data = raw_data['image'].reshape(data_size, 784)        # 12000*(28*28) images
    label = utils.to_categorical(raw_data['label'], 10)     # 12000*10

    lable_count = list(np.sum(label, 0))
    label_list = np.argmax(label, axis=1)

    # plt.figure(data_type + " distribution",figsize=(10,5))
    # plt.title(data_type + " distribution")
    # plt.bar(*np.unique(label_list, return_counts=True))
    # plt.show()

    return data, label

def data_shuffling(train, target):
    assert len(train) == len(target), "Length of train and target are not the same"
    
    concate_c = np.c_[train.reshape(len(train), -1), target.reshape(len(target), -1)]
    np.random.shuffle(concate_c)
    s_train = concate_c[:, :train.size//len(train)].reshape(train.shape)
    s_target = concate_c[:, train.size//len(train):].reshape(target.shape)

    return s_train, s_target

def create_linear_layer(input_size, output_size, use_bias=False, use_zeros_w=False):
    # weight = np.zeros((input_size, output_size))
    if not use_zeros_w:
        weight = np.random.randn(input_size, output_size)
    else:
        weight = np.zeros((input_size, output_size))
    
    bias = np.zeros(output_size) #(output_size, )
    # bias = np.zeros((output_size, 1))   #(output_size, 1)
    # print(weight.shape, bias.shape)
    return weight, bias

def cal_loss(predict, gt):
    # cross_entropy, loss for single data
    gt_cls_index = list(gt).index(1)
    # print("gt", gt)
    # print("predict",predict)
    pred_cls = predict[gt_cls_index]
    if pred_cls < 0.001:
        pred_cls = 0.001

    return -1*math.log(pred_cls)

def forward_pass(x, y, batch_param, back_prop=True, use_latent_nodes=False):
    w1 = batch_param["w1"].copy()
    w2 = batch_param["w2"].copy()
    w3 = batch_param["w3"].copy()
    b1 = batch_param["b1"].copy()
    b2 = batch_param["b2"].copy()
    b3 = batch_param["b3"].copy()
    w4 = batch_param["w4"].copy()
    b4 = batch_param["b4"].copy()

    if use_latent_nodes:
        x = normailize(x)
        z1 = np.dot(x, w1) + b1         #x:(748, ) y:(10, )      #z1:(100,)
        a1 = LeakyReLU(z1)               #
        z2 = np.dot(a1, w2) + b2
        a2 = LeakyReLU(z2)
        z3 = np.dot(a2, w3) + b3        #z3:(10,)
        a3 = LeakyReLU(z3)
        z4 = np.dot(a3, w4) + b4
        a4 = softmax(z4)

        # cal cost
        single_loss = cal_loss(a4, y)
        cache = {"z1": z1, "z2": z2, "z3": z3, "a1": a1, "a2": a2, "a3": a3, "z4": z4, "a4": a4}
        
        # back_pass
        grad_param = {}
        if back_prop == True:
            grad_param = back_pass(x, y, cache, batch_param, use_latent_nodes)
    
    else:
        x = normailize(x)
        z1 = np.dot(x, w1) + b1         #x:(748, ) y:(10, )      #z1:(100,)
        # a1 = ReLu(z1)                   
        a1 = LeakyReLU(z1)                   
        z2 = np.dot(a1, w2) + b2
        # a2 = ReLu(z2)
        a2 = LeakyReLU(z2)
        z3 = np.dot(a2, w3) + b3        #z3:(10,)
        a3 = softmax(z3)


        # cal cost
        single_loss = cal_loss(a3, y)
        cache = {"z1": z1, "z2": z2, "z3": z3, "a1": a1, "a2": a2, "a3": a3}
        
        # back_pass
        grad_param = {}
        if back_prop == True:
            grad_param = back_pass(x, y, cache, batch_param, use_latent_nodes)
    
    return grad_param, cache, single_loss

def back_pass(x, y, cache, parameters, use_latent_nodes=False):
    z1 = np.reshape(cache["z1"].copy(), (cache["z1"].shape[0],1))
    z2 = np.reshape(cache["z2"].copy(), (cache["z2"].shape[0],1))
    z3 = np.reshape(cache["z3"].copy(), (cache["z3"].shape[0],1))
    a1 = np.reshape(cache["a1"].copy(), (z1.shape[0],1))
    a2 = np.reshape(cache["a2"].copy(), (z2.shape[0],1))
    a3 = np.reshape(cache["a3"].copy(), (z3.shape[0],1))
    w1 = parameters["w1"].copy()
    w2 = parameters["w2"].copy()
    w3 = parameters["w3"].copy()
    b1 = parameters["b1"].copy()
    b2 = parameters["b2"].copy()
    b3 = parameters["b3"].copy()

    label = np.reshape(y.copy(), (y.shape[0],1))
    train = np.reshape(x.copy(), (x.shape[0],1))

    # layer 3 to 2
    grad_z1 = np.zeros((z1.shape[0], 1))  # (100,)
    grad_z2 = np.zeros((z2.shape[0], 1))  # (500,)
    grad_z3 = np.zeros((z3.shape[0], 1))  # (10,)

    grad_w1 = np.zeros((w1.shape))  # (784, 100)
    grad_w2 = np.zeros((w2.shape))  # (100,  50)
    grad_w3 = np.zeros((w3.shape))  # ( 50,  10)
    grad_b1 = np.zeros((b1.shape[0], 1))  # (100,1)
    grad_b2 = np.zeros((b2.shape[0], 1))  # (50, 1)
    grad_b3 = np.zeros((b3.shape[0], 1))  # (10, 1)


    if use_latent_nodes:
        # using latent, additional layer
        z4 = np.reshape(cache["z4"].copy(), (cache["z4"].shape[0],1))
        a4 = np.reshape(cache["a4"].copy(), (z4.shape[0],1))
        w4 = parameters["w4"].copy()
        b4 = parameters["b4"].copy()
        grad_z4 = np.zeros((z4.shape[0], 1))  # (10,)
        grad_w4 = np.zeros((w4.shape))  # ( 2,  10)
        grad_b4 = np.zeros((b4.shape[0], 1))  # (10, 1)

        # softmax
        grad_z4 = a4 - label
        grad_w4 = np.dot(a3, grad_z4.T)
        grad_b4 = grad_z4.copy()

        # ReLU
        grad_a3 = np.dot(w4, grad_z4)
        # sign_matrix = np.sign(z3)
        # sign_matrix[sign_matrix>=0] = 1
        # sign_matrix[sign_matrix<0] = 0
        # print(a3)
        # print(z3)
        # print(sign_matrix)
        # ReLU grad
        # sign_matrix_old = np.sign(np.multiply(a3, z3))
        # print("old sign:\n", sign_matrix_old)
        # assert (sign_matrix_old == sign_matrix).all()
        # exit(-1)
        # LeakyReLU
        grad_leaky_a_z = np.sign(z3)
        grad_leaky_a_z[grad_leaky_a_z>=0] = 1
        grad_leaky_a_z[grad_leaky_a_z<0] = leaky_alpha
        # grad_z3 = np.multiply(grad_a3, sign_matrix_old)
        grad_z3 = np.multiply(grad_a3, grad_leaky_a_z)
        grad_w3 = np.dot(a2, grad_z3.T)
        grad_b3 = grad_z3.copy()

        # ReLU
        grad_a2 = np.dot(w3, grad_z3)
        grad_leaky_a_z = np.sign(z2)
        grad_leaky_a_z[grad_leaky_a_z>=0] = 1
        grad_leaky_a_z[grad_leaky_a_z<0] = leaky_alpha
        grad_z2 = np.multiply(grad_a2, grad_leaky_a_z)
        grad_w2 = np.dot(a1, grad_z2.T)
        grad_b2 = grad_z2.copy()

        # ReLU
        grad_a1 = np.dot(w2, grad_z2)
        grad_leaky_a_z = np.sign(z1)
        grad_leaky_a_z[grad_leaky_a_z>=0] = 1
        grad_leaky_a_z[grad_leaky_a_z<0] = leaky_alpha
        grad_z1 = np.multiply(grad_a1, grad_leaky_a_z)
        grad_w1 = np.dot(train, grad_z1.T)
        grad_b1 = grad_z1.copy()

        grad_param = {"g_w1": grad_w1, "g_w2": grad_w2, "g_w3": grad_w3, \
                    "g_b1": grad_b1, "g_b2": grad_b2, "g_b3": grad_b3, \
                    "g_w4": grad_w4, "g_b4": grad_b4}
    
    else:    
        # softmax
        grad_z3 = a3 - label
        grad_w3 = np.dot(a2, grad_z3.T)
        grad_b3 = grad_z3.copy()

        # ReLU
        grad_a2 = np.dot(w3, grad_z3)
        # sign_matrix = np.sign(np.multiply(a2, z2))
        # grad_z2 = np.multiply(grad_a2, sign_matrix)
        grad_leaky_a_z = np.sign(z2)
        grad_leaky_a_z[grad_leaky_a_z>=0] = 1
        grad_leaky_a_z[grad_leaky_a_z<0] = leaky_alpha
        grad_z2 = np.multiply(grad_a2, grad_leaky_a_z)
        grad_w2 = np.dot(a1, grad_z2.T)
        grad_b2 = grad_z2.copy()

        # ReLU
        grad_a1 = np.dot(w2, grad_z2)
        # sign_matrix = np.sign(np.multiply(a1, z1))
        # grad_z1 = np.multiply(grad_a1, sign_matrix)
        grad_leaky_a_z = np.sign(z1)
        grad_leaky_a_z[grad_leaky_a_z>=0] = 1
        grad_leaky_a_z[grad_leaky_a_z<0] = leaky_alpha
        grad_z1 = np.multiply(grad_a1, grad_leaky_a_z)
        grad_w1 = np.dot(train, grad_z1.T)
        grad_b1 = grad_z1.copy()

        grad_param = {"g_w1": grad_w1, "g_w2": grad_w2, "g_w3": grad_w3, \
                    "g_b1": grad_b1, "g_b2": grad_b2, "g_b3": grad_b3}
    
    return grad_param

def updata_parameter(parameters, w1, w2, w3, b1, b2, b3, w4, b4):
    # print("before update: ")
    # print(np.sum(parameters["w1"]))
    parameters["w1"] = w1
    parameters["w2"] = w2
    parameters["w3"] = w3
    parameters["b1"] = b1
    parameters["b2"] = b2
    parameters["b3"] = b3
    parameters["w4"] = w4
    parameters["b4"] = b4
    # print("after:")
    # print(np.sum(parameters["w1"]))
    
def cal_error_rate(data, label, parameters, use_latent_nodes=False):
    assert len(data) == len(label)

    confusion_matrix = np.zeros((label.shape[1], label.shape[1]))
    # print(confusion_matrix.shape)
    final_param = parameters.copy()
    
    for x,y in zip(data, label):
        _, cache, _ = forward_pass(x, y, final_param, back_prop=False, use_latent_nodes=use_latent_nodes)
        if use_latent_nodes:
            output_layer = "a4"
        else:
            output_layer = "a3"
        
        update_confusion(cache[output_layer], y, confusion_matrix)
        
    # print(confusion_matrix)
    # print(np.sum(confusion_matrix))
    assert np.sum(confusion_matrix) == len(data)

    return (1-np.trace(confusion_matrix)/len(data)), confusion_matrix

def update_confusion(predict, gt, confusion_matrix):
    pred_cls_idx = np.argmax(predict)
    gt_cls_idx = list(gt).index(1)
    confusion_matrix[pred_cls_idx][gt_cls_idx] += 1

def ReLu(z):
    # print(z)
    # print(np.maximum(z ,np.zeros(len(z))))
    return np.maximum(z ,np.zeros(len(z)))

def LeakyReLU(z):
    a = z.copy()
    # print(z)
    # print(z > 0)
    a[a<0] *= leaky_alpha
    # print(z) 
    return a

def softmax(z):
    # print("final softmax")
    # print(np.min(z), np.max(z))
    # print(z)
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    # print(np.exp(z) / np.sum(np.exp(z)))
    # print(np.exp(z_shift) / np.sum(np.exp(z_shift)))
    return exp_z / np.sum(exp_z)

def normailize(x):
    return x/np.max(x)

def output_figure(iteration_loss_list, epoch_loss_list, train_error_rate_list, test_error_rate_list, current_epoch, epoch_file_dir, zero_initial):

    # plot iteration loss
    fig1 = plt.figure("Iteration loss",figsize=(10,5))
    plt.title("Iteration loss")
    plt.xlabel('Iteration',fontsize=14)
    plt.ylabel('Ave loss',fontsize=14)
    plt.grid(True)
    plt.plot(range(1, len(iteration_loss_list)+1), iteration_loss_list, linestyle='-', zorder = 1, color = 'blue', linewidth=1)
    plt.savefig(os.path.join(epoch_file_dir, "Iter loss_"+str(current_epoch)+'_'+str(zero_initial)))

    fig2 = plt.figure("Epoch loss" ,figsize=(10,5))
    plt.title("Epoch loss ("+str(epoch_loss_list[-1])+")")
    plt.xlabel('Epoch',fontsize=14)
    plt.ylabel('Ave loss',fontsize=14)
    plt.grid(True)
    plt.plot(range(1, len(epoch_loss_list)+1), epoch_loss_list, linestyle='-', zorder = 1, color = 'red', linewidth=1)
    plt.savefig(os.path.join(epoch_file_dir, "Epoch loss_"+str(current_epoch)+'_'+str(zero_initial)))
    
    # error rate
    fig3 = plt.figure("Train error rate",figsize=(10,5))
    plt.title("Train error rate ("+str(train_error_rate_list[-1])+")")
    plt.xlabel('Epoch',fontsize=14)
    plt.ylabel('Error rate',fontsize=14)
    plt.grid(True)
    plt.plot(range(1, len(train_error_rate_list)+1), train_error_rate_list, linestyle='-', zorder = 1, color = 'blue', linewidth=1)
    plt.savefig(os.path.join(epoch_file_dir, "Train error_"+str(current_epoch)+'_'+str(zero_initial)))

    fig4 = plt.figure("Test error rate",figsize=(10,5))
    plt.title("Test error rate ("+str(test_error_rate_list[-1])+")")
    plt.xlabel('Epoch',fontsize=14)
    plt.ylabel('Error rate',fontsize=14)
    plt.grid(True)
    plt.plot(range(1, len(test_error_rate_list)+1), test_error_rate_list, linestyle='-', zorder = 1, color = 'blue', linewidth=1)
    plt.savefig(os.path.join(epoch_file_dir, "Test error_"+str(current_epoch)+'_'+str(zero_initial)))
    # plt.show()

    fig1.clear()
    fig2.clear()
    fig3.clear()
    fig4.clear()

def output_log(epoch_file_dir, hidden_node_num, epoch_num, zero_initial, iteration_num, epoch_loss_list, train_error_rate_list, test_error_rate_list, train_confu, test_confu, epoch_weight, use_latent_nodes=False):
    
    with open(os.path.join(epoch_file_dir, str(epoch_num)+ '_' + str(zero_initial) + '_weight.txt'),'w') as f:
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write("Time: {}\n".format(current_time))
        f.write("Batch size: {}\n".format(batch_size))
        f.write("Epoch num: {}\n".format(epoch_num))
        f.write("Learning rate: {}\n".format(learning_rate))
        f.write("Leaky ReLU alpha: {}\n".format(leaky_alpha))
        f.write("Iteration per epoch: {}\n".format(iteration_num))
        f.write("Final loss: {:.5f}\n".format(epoch_loss_list[-1]))
        f.write("Train error rate: {:.5f}\n".format(train_error_rate_list[-1]))
        f.write("Test error rate: {:.5f}\n".format(test_error_rate_list[-1]))
        f.write("Hidden nodes:")
        np.savetxt(f, np.reshape(np.array(hidden_node_num), (1, len(hidden_node_num))), fmt='%d')
        f.write("\n\nInitial weight:\n")
        param_0 = epoch_weight[0]
        w1_0 = param_0["w1"]
        w2_0 = param_0["w2"]
        w3_0 = param_0["w3"]
        b1_0 = param_0["b1"]
        b2_0 = param_0["b2"]
        b3_0 = param_0["b3"]

        np.savetxt(f, w1_0, fmt='%.3f', header='w1_0')
        np.savetxt(f, w2_0, fmt='%.3f', header='w2_0')
        np.savetxt(f, w3_0, fmt='%.3f', header='w3_0')
        np.savetxt(f, b1_0, fmt='%.3f', header='b1_0')
        np.savetxt(f, b2_0, fmt='%.3f', header='b2_0')
        np.savetxt(f, b3_0, fmt='%.3f', header='b3_0')

        if use_latent_nodes:
            w4_0 = param_0["w4"]
            b4_0 = param_0["b4"]
            np.savetxt(f, w4_0, fmt='%.3f', header='w4_0')
            np.savetxt(f, b4_0, fmt='%.3f', header='b4_0')

        f.write("\n\nFinal weight:\n")
        param = epoch_weight[-1]
        w1 = param["w1"]
        w2 = param["w2"]
        w3 = param["w3"]
        b1 = param["b1"]
        b2 = param["b2"]
        b3 = param["b3"]

        np.savetxt(f, w1, fmt='%.3f', header='w1')
        np.savetxt(f, w2, fmt='%.3f', header='w2')
        np.savetxt(f, w3, fmt='%.3f', header='w3')
        np.savetxt(f, b1, fmt='%.3f', header='b1')
        np.savetxt(f, b2, fmt='%.3f', header='b2')
        np.savetxt(f, b3, fmt='%.3f', header='b3')

        if use_latent_nodes:
            w4 = param["w4"]
            b4 = param["b4"]
            np.savetxt(f, w4, fmt='%.3f', header='w4')
            np.savetxt(f, b4, fmt='%.3f', header='b4')
        
        np.savetxt(f, train_confu, fmt='%d', header='train_confusion')
        np.savetxt(f, test_confu, fmt='%d', header='test_confusion')
        f.close()

def cal_latent_features(s_data, s_label, weight, epoch_file_dir, current_epoch):
    assert len(s_data) == len(s_label)
    labels = []
    f_x = []
    f_y = []
    for i in range(10):
        f_x.append([])
        f_y.append([])
    for x, y in zip(s_data, s_label):
        _, cache, _ = forward_pass(x, y, weight, back_prop=False, use_latent_nodes=True)
        labels.append(list(y).index(1))
        f_x[list(y).index(1)].append(cache["z3"][0])
        f_y[list(y).index(1)].append(cache["z3"][1])

    fig = plt.figure("2D feature "+str(current_epoch)+" epoch",figsize=(10,5))
    for l in range(10):
        plt.scatter(f_x[l],f_y[l],label=l)

    plt.legend()
    plt.title("2D feature of "+str(current_epoch)+" epoch")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.savefig(os.path.join(epoch_file_dir, "feature_"+str(current_epoch))) 
    fig.clear()

def make_output_dir(total_epoch_num):
    epoch_file_dir = os.path.join(fig_output_dir, "epoch_"+str(total_epoch_num)+"_"+str(leaky_alpha))
    if not os.path.exists(epoch_file_dir):
        os.makedirs(epoch_file_dir)
    return epoch_file_dir

def output_weight(epoch_file_dir, parameters, epoch_num, use_latent_nodes):
    weight_file_dir = os.path.join(epoch_file_dir, "w")
    if not os.path.exists(weight_file_dir):
        os.makedirs(weight_file_dir)
    
    w_file = os.path.join(weight_file_dir, str(epoch_num)+"_weights.npz")
    if not use_latent_nodes:
        np.savez(w_file, \
                w1=parameters["w1"], \
                w2=parameters["w2"], \
                w3=parameters["w3"], \
                b1=parameters["b1"], \
                b2=parameters["b2"], \
                b3=parameters["b3"])
    else:
        np.savez(w_file, \
                w1=parameters["w1"], \
                w2=parameters["w2"], \
                w3=parameters["w3"], \
                w4=parameters["w4"], \
                b1=parameters["b1"], \
                b2=parameters["b2"], \
                b3=parameters["b3"], \
                b4=parameters["b4"])


if __name__ =="__main__":
    
    # # make every random seed as the same situation for everty time
    # np.random.seed(1)    

    data, label = load_data()
    test_data, test_label = load_data(data_type="test")
    
    
    # print(type(data[0]))
    # cv2.imshow("train_0", np.reshape(data[0], (28, 28)))
    # cv2.waitKey()

    # epoch_num_list = [[500, True], \
    # [500, False], \
    # [700, True], \
    # [700, False]]

    # epoch_num_list = [[5, False], [6, False]]

    epoch_num_list = [[1002, False], \
    [1003, False]]

    use_latent_nodes = True

    # for epoch_num in epoch_num_list:
    for epoch_num, zero_initial in epoch_num_list:
        layer_num = 3
        hidden_node_num = [100, 50, 10]
        # hidden_node_num = [100, 2, 10]

        if use_latent_nodes == True:
            # add additional layer of 2 to visualize output features
            hidden_node_num = [100, 50, 2, 10]

        w1, b1 = create_linear_layer(data.shape[1], hidden_node_num[0], use_zeros_w=zero_initial)
        w2, b2 = create_linear_layer(hidden_node_num[0], hidden_node_num[1], use_zeros_w=zero_initial)
        w3, b3 = create_linear_layer(hidden_node_num[1], hidden_node_num[2], use_zeros_w=zero_initial)
        w4, b4 = create_linear_layer(hidden_node_num[-2], hidden_node_num[-1], use_zeros_w=zero_initial)
        parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3, "w4": w4, "b4": b4}

        totol_sample_num = len(data)
        iteration_num = int(totol_sample_num / batch_size)
        print("Total epoch #: {}".format(epoch_num))
        print("Zero w: {}".format(zero_initial))
        print("Batch size: {}".format(batch_size)) 
        print("Iteration #/epoch: {}".format(iteration_num))
        print("Hidden layer: {}".format(hidden_node_num))
        print("Start training....\n\n")

        iteration_loss_list = []
        epoch_loss_list = []
        train_error_rate_list = []
        test_error_rate_list = []
        epoch_weight = []
        train_confu = np.zeros((label.shape[1],label.shape[1]))
        test_confu = np.zeros((label.shape[1],label.shape[1]))

        epoch_file_dir = make_output_dir(epoch_num)

        epoch_progress = tqdm(total=epoch_num)
        for i in range(epoch_num):
            tqdm.write("Epoch {}".format(i))
            s_data, s_label = data_shuffling(data, label)

            w1 = parameters["w1"].copy()
            w2 = parameters["w2"].copy()
            w3 = parameters["w3"].copy()
            b1 = parameters["b1"].copy()
            b2 = parameters["b2"].copy()
            b3 = parameters["b3"].copy()
            w4 = parameters["w4"].copy()
            b4 = parameters["b4"].copy()

            epoch_weight.append(parameters)

            # iteration_loss_list = []
            epoch_total_loss = 0

            # mini-batch SGD, using mini batch to update parameter once/(1 iteration)
            # cal each sample's gradient's by forward/backward in each batch, them average all gradient in batch and update param
            for j in range(iteration_num):
                # print("Iter {}".format(j))
                # print(np.sum(parameters["w1"])
                x_bacth = s_data[j*batch_size:(j+1)*batch_size, :] #200*748
                y_batch = s_label[j*batch_size:(j+1)*batch_size, :] #200*10
                batch_param =  parameters.copy()
                # print("Using new param: {}".format(np.sum(batch_param["w1"])))

                # sum of grad for each data in min-batch
                w1_grad_sum, b1_grad_sum = create_linear_layer(w1.shape[0], w1.shape[1], use_zeros_w=True)
                w2_grad_sum, b2_grad_sum = create_linear_layer(w2.shape[0], w2.shape[1], use_zeros_w=True)
                w3_grad_sum, b3_grad_sum = create_linear_layer(w3.shape[0], w3.shape[1], use_zeros_w=True)
                w4_grad_sum, b4_grad_sum = create_linear_layer(w4.shape[0], w4.shape[1], use_zeros_w=True)

                sample_count = 0
                batch_loss = 0

                for x, y in zip(x_bacth, y_batch):
                    grad_param, _, single_loss = forward_pass(x, y, batch_param, use_latent_nodes=use_latent_nodes)
                    batch_loss += single_loss

                    # accumulate grad_param for mini_batch data, then average them before update at iteration cycle
                    w1_grad_sum += grad_param["g_w1"]
                    w2_grad_sum += grad_param["g_w2"]
                    w3_grad_sum += grad_param["g_w3"]
                    b1_grad_sum += grad_param["g_b1"].reshape(-1)
                    b2_grad_sum += grad_param["g_b2"].reshape(-1)
                    b3_grad_sum += grad_param["g_b3"].reshape(-1)

                    if use_latent_nodes:
                        # tqdm.write("update")
                        # tqdm.write(str(np.sum(grad_param["g_w4"])))
                        w4_grad_sum += grad_param["g_w4"]
                        b4_grad_sum += grad_param["g_b4"].reshape(-1)

                    sample_count += 1
                    # tqdm.write(str(np.sum(w4_grad_sum)))
                    # print(np.sum(grad_param["g_w1"]))
                    # print(np.sum(grad_param["g_w2"]))
                    # print(np.sum(grad_param["g_w3"]))

                # finish mini-batch, update by average gradient
                w1 = batch_param["w1"] - learning_rate * (1/batch_size) * w1_grad_sum
                w2 = batch_param["w2"] - learning_rate * (1/batch_size) * w2_grad_sum
                w3 = batch_param["w3"] - learning_rate * (1/batch_size) * w3_grad_sum
                b1 = batch_param["b1"] - learning_rate * (1/batch_size) * b1_grad_sum
                b2 = batch_param["b2"] - learning_rate * (1/batch_size) * b2_grad_sum
                b3 = batch_param["b3"] - learning_rate * (1/batch_size) * b3_grad_sum
                # if no use_latent_feature, would remain zeros
                w4 = batch_param["w4"] - learning_rate * (1/batch_size) * w4_grad_sum
                b4 = batch_param["b4"] - learning_rate * (1/batch_size) * b4_grad_sum

                
                # update param to use new ones iteration at next batch
                updata_parameter(parameters, w1, w2, w3, b1, b2, b3, w4, b4)

                # record iteration loss
                iteration_loss_list.append(batch_loss/batch_size)

                epoch_total_loss += (batch_loss/batch_size)

                # print("{} iteration".format(j))
                # print(np.sum(parameters["w1"]))

            # check error rate
            train_rate, train_confu = cal_error_rate(data, label, parameters, use_latent_nodes)
            test_rate, test_confu = cal_error_rate(test_data, test_label, parameters, use_latent_nodes)
            train_error_rate_list.append(train_rate)
            test_error_rate_list.append(test_rate)
            tqdm.write("train error rate: {:.3f}".format(train_rate))
            tqdm.write("test error rate: {:.3f}".format(test_rate))

            # calculate epoch loss, average all batch loss
            epoch_loss_list.append(epoch_total_loss/iteration_num)
            epoch_progress.update(1)
            tqdm.write("Ave Epoch Loss: {:.3f}".format(epoch_loss_list[-1]))        
            tqdm.write('-'*40)

            # render figure
            if i % 100 == 0 and i != 0:
                output_figure(
                    iteration_loss_list, \
                    epoch_loss_list, \
                    train_error_rate_list, \
                    test_error_rate_list, \
                    i, \
                    epoch_file_dir, \
                    zero_initial)

            # render latent layer features
            if use_latent_nodes == True and (i == 200 or i == 500 or i == 800 or i == epoch_num-1):
                cal_latent_features(test_data, test_label, epoch_weight[-1], epoch_file_dir, i)

            output_weight(epoch_file_dir, parameters, epoch_num, use_latent_nodes)


        # output epoch weight
        output_log(
            epoch_file_dir, \
            hidden_node_num, \
            epoch_num, \
            zero_initial, \
            iteration_num, \
            epoch_loss_list, \
            train_error_rate_list, \
            test_error_rate_list, \
            train_confu, \
            test_confu, \
            epoch_weight, \
            use_latent_nodes)
    
        # render figure
        output_figure(
            iteration_loss_list, \
            epoch_loss_list, \
            train_error_rate_list, \
            test_error_rate_list, \
            epoch_num, \
            epoch_file_dir, \
            zero_initial)

        print("End train total {} epochs".format(epoch_num))




