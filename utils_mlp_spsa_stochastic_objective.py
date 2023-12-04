import os
import os.path
import datetime
import time
import numpy as np
import torch.nn as nn
from scipy import*
from copy import*
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import shutil
from tqdm import tqdm
import torch

def initDataframe_pretraining(path, dataframe_to_init = 'pre_training.csv'):
    '''
    Initialize a dataframe with Pandas so that the pre-training loss is stored
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep = ',', index_col = 0)
    else:
        columns_header = ['Pre-training loss' ]

        dataframe = pd.DataFrame({},columns = columns_header)
        dataframe.to_csv(path + prefix + dataframe_to_init)

    return dataframe


def updateDataframe_pretraining(BASE_PATH, pretraining_loss, dataframe_to_update = 'pre_training.csv'):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    data = [pretraining_loss]

    dataframe = pd.read_csv(BASE_PATH + prefix + dataframe_to_update, sep = ',', index_col = 0) #load old dataframe
    new_data = pd.DataFrame([data],index=[1],columns=dataframe.columns) #create new one

    dataframe = pd.concat([dataframe, new_data], axis=0) #concat both
    dataframe.to_csv(BASE_PATH + prefix + dataframe_to_update)

    return dataframe


def save_Dataframe_classifier(path, data, dataframe_to_init = 'classifier.csv'):
    '''
    Initialize a dataframe with Pandas so that the pre-training loss is stored
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep = ',', index_col = 0)
    else:
        columns_header = ['Training loss', 'Testing loss', 'Training error', 'Testing error']
        data = np.array(data)

        dataframe = pd.DataFrame(data.T, columns = columns_header)
        dataframe.to_csv(path + prefix + "Classifiers" + prefix + dataframe_to_init)

    return dataframe


def updateDataframe_classifier(BASE_PATH, datas, dataframe_to_update = 'classifier.csv'):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
   
    data = [datas[0], datas[1], datas[2], datas[3]] #train loss, test loss, train error, test error

    dataframe = pd.read_csv(BASE_PATH + prefix + "Classifiers" + prefix + dataframe_to_update, sep = ',', index_col = 0) #load old dataframe
    new_data = pd.DataFrame([data],index=[1],columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)
    dataframe.to_csv(BASE_PATH + prefix + dataframe_to_update)

    return dataframe


def createPath(archi = "MLP", dataset = "MNIST"):
    '''
    Create path to save data
    '''

    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    BASE_PATH += prefix + 'DATA-' + archi + "-" + dataset

    BASE_PATH += prefix + datetime.datetime.now().strftime("%Y-%m-%d")


    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    _ = shutil.copy('plotFunction.py', BASE_PATH)
    #filePath = shutil.copy('plot-notebook.ipynb', BASE_PATH)
    
    files = os.listdir(BASE_PATH)

    if 'plotFunction.py' in files:
        files.pop(files.index('plotFunction.py'))

    if not files:
        BASE_PATH = BASE_PATH + prefix + 'S-1'
    else:
        tab = []
        if '.DS_Store' in files:
            files.pop(files.index('.DS_Store'))
        for names in files:
            tab.append(int(names.split('-')[1]))
        BASE_PATH += prefix + 'S-' + str(max(tab)+1)

    try:
        os.mkdir(BASE_PATH)
    except:
        pass
    
    _ = shutil.copy('plot-notebook.ipynb', BASE_PATH)
    
    try:
        os.mkdir(BASE_PATH + prefix + "Weights")
    except:
        pass
    
    try:
        os.mkdir(BASE_PATH + prefix + "Models")
    except:
        pass
    
    try:
        os.mkdir(BASE_PATH + prefix + "Classifiers")
    except:
        pass

    return BASE_PATH


def saveHyperparameters(args, BASE_PATH):
    '''
    Save all hyperparameters in the path provided
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    f = open(BASE_PATH + prefix + 'Hyperparameters.txt', 'w')
    f.write('Layer-Wise SSL \n')
    f.write('   Parameters of the simulation \n ')
    f.write('\n')

    for key in args.__dict__:
        f.write(key)
        f.write(': ')
        f.write(str(args.__dict__[key]))
        f.write('\n')

    f.close()
    
    
def save_weights(BASE_PATH, prefix, net, epoch):
    '''
    Store the first 100 weights of the first weight matrix
    '''
    fig, axs = plt.subplots(10, 10, figsize=(5, 5))

    for i, ax in enumerate(axs.flat):
        ax.imshow(net.layers[0].weight[i,:].view(28,28).detach().cpu(), cmap= "gray")  # Plot the weight matrix
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(BASE_PATH + prefix + "Weights" + prefix + "Weights_epoch#" + str(epoch) + ".pdf", format = "pdf")
    plt.close()
    
    return 0


def store_checkpoint(BASE_PATH, args, net, epoch, layer, loss):
    '''
    Function that store a checkpoint after each epoch of pretraining
    Store the checkpoint of the model
    Store the current pretraining loss
    Store the 
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
                     
    #store checkpoint                
    torch.save(net.state_dict(), BASE_PATH + prefix + "Models" + prefix + "checkpoint.pt")
    
    #store weight at some epochs
    if (epoch % 20) == 0 and layer == 0:
        save_weights(BASE_PATH, prefix, net, epoch)
                     
    #store the pretraining loss              
    updateDataframe_pretraining(BASE_PATH, loss, dataframe_to_update = 'pre_training.csv')
    
    return 0  
                     
    
def pretrain(args, net, train_loader, train_layer = 0):
    '''
    Pre-train the network for 1 epoch
    Train_layer indicates which layer to specifically train (sequential training)
    '''
    objs = ['sim', 'std', 'cov'] #list of the 3 objectives to minimize
    net.eval()
    loss_tot = 0
    with torch.no_grad():
        comp_mini_batch = 0
        for batch_idx, (datas, _) in enumerate(tqdm(train_loader, position = 0, leave = True)): #now datas has a len of n_average!
            #print("-------")
            #0. for this mini-batch - do a random choise for the objective to optimize - uniform probability for the 3? or should we include the vicreg params in the probability here??
            obj_loc = np.random.choice(objs, 1).item()
            net.optimizer.zero_grad()
            
            for idx, data in enumerate(datas):
                #1. get sub-mini batch data
                data = net.single_batch(data) #set the first dim to be n_views*batch_size
                data = data.to(net.device) #put on the GPU
                
                #2. generate a random pertubation matrix for the weights
                perturbation = net.generate_perturbation(layer = train_layer) #generate random perturbation
                #print(perturbation)
                
                #3. compute the two forward passes
                pos_obj_repr, pos_obj_std, pos_obj_cov = net(data, train_layer, perturbation) #the loss is computed at each layer in the forward function
                neg_obj_repr, neg_obj_std, neg_obj_cov = net(data, train_layer, -1*perturbation) #the loss is computed at each layer in the forward function
                pos_obj = net.losses[train_layer].sim_coeff*pos_obj_repr + net.losses[train_layer].std_coeff* pos_obj_std + net.losses[train_layer].cov_coeff*pos_obj_cov
                
                if idx == 0: #for the first "mini-mini-batch" we compute the gradient
                    if obj_loc == 'sim':
                        grads = net.compute_spsa(perturbation, pos_obj_repr, neg_obj_repr, layer = train_layer)
                    elif obj_loc == 'std':
                        grads = net.compute_spsa(perturbation, pos_obj_std, neg_obj_std, layer = train_layer)
                    elif obj_loc == 'cov':
                        grads = net.compute_spsa(perturbation,  pos_obj_cov, neg_obj_cov, layer = train_layer)
                        
                else:
                    if obj_loc == 'sim':
                        grads += net.compute_spsa(perturbation, pos_obj_repr, neg_obj_repr, layer = train_layer)
                    elif obj_loc == 'std':
                        grads += net.compute_spsa(perturbation, pos_obj_std, neg_obj_std, layer = train_layer)
                    elif obj_loc == 'cov':
                        grads += net.compute_spsa(perturbation,  pos_obj_cov, neg_obj_cov, layer = train_layer)
                        
            grads /= args.n_average #average the resulting gradient
            
            if obj_loc == 'sim':
                grads *= net.losses[train_layer].sim_coeff
            elif obj_loc == 'std':
                grads *= net.losses[train_layer].std_coeff
            elif obj_loc == 'cov':
                grads *= net.losses[train_layer].cov_coeff
                  
            net.optimizer.zero_grad()
            net.apply_spsa(grads, layer = train_layer) #apply the computed grad to the corresponding parameters of the network
            
            loss_tot += pos_obj.item()
            net.optimizer.step() #optimizer step with the gradients we fed to the parameters

    return net, loss_tot/len(train_loader.dataset)


def pretraining_loop(BASE_PATH, args, net, train_loader, train_loader_classifier, test_loader, epochs = 20):
    '''
    pre-train the MLP for N epochs
    '''
    loss_tot = []
    store_checkpoint(BASE_PATH, args, net, -1, 0,  0)
    
    for layer in range(args.nlayers):
        net.reset_perturbation_step() #reset the perturbation step to 1
        for epoch in tqdm(range(epochs), position = 0, leave = True):
            net, loss = pretrain(args, net, train_loader, train_layer = layer)
            loss_tot.append(loss)
            # ADD training linear classifier every k epochs
            
            store_checkpoint(BASE_PATH, args, net, epoch, layer,  loss)

            if epoch%50 == 0: #train linear classifiers every 50 epochs to see the convergence
                print("training linear classifier")
                training_classifiers(BASE_PATH, args, net, epoch, train_loader_classifier, test_loader, layer = layer)
        net.update_perturbation_step() #decay the perturbation step
        
    return net, loss_tot  
                     

def scaled_sigmoid(x, a, b, s, z):
    return s/(1+np.exp(-a*(x-b)))+z


def param_sigmoid(args, plot = True):
    '''
    Function to generate the parametrization (sigmoid) coefficients
    '''
    
    colormap = plt.cm.Reds

    colors = [colormap(i) for i in np.linspace(0.2,1,4)]

    scale = args.scale
    slope = args.slope
    threshold = args.threshold
    bias = args.bias

    #continuous settings (visualization only!)
    x = np.linspace(1,4, 100)
    ysim = scaled_sigmoid(x, slope[0], threshold[0], scale[0], bias[0])
    yvar =  scaled_sigmoid(x, slope[1], threshold[1], scale[1], bias[1])
    ycovar =  scaled_sigmoid(x, slope[2], threshold[2], scale[2], bias[2])
    yintravar =  scaled_sigmoid(x, slope[3], threshold[3], scale[3], bias[3])

    if plot == True:
        plt.figure()
        plt.plot(x, ysim, label = "similarity", color = colors[3])
        plt.plot(x, yvar, label = "variance", color = colors[2])
        plt.plot(x, ycovar, label = "covariance", color = colors[1])
        plt.plot(x, yintravar, label = "intra-sample variance", color = colors[0])

    # discrete settings
    x = np.arange(1,5)
    ysim = scaled_sigmoid(x, slope[0], threshold[0], scale[0], bias[0])
    yvar =  scaled_sigmoid(x, slope[1], threshold[1], scale[1], bias[1])
    ycovar =  scaled_sigmoid(x, slope[2], threshold[2], scale[2], bias[2])
    yintravar =  scaled_sigmoid(x, slope[3], threshold[3], scale[3], bias[3])

    if plot == True:                 
        plt.plot(x, ysim, "ko", label = "similarity")
        plt.plot(x, yvar, "ko", label = "variance")
        plt.plot(x, ycovar, "ko", label = "covariance")
        plt.plot(x, yintravar, "ko", label = "intra-sample variance")


        plt.ylim([0,30])
        plt.legend(bbox_to_anchor=(1.1, 1))
        plt.show()

    print("ysim = " + str(ysim))
    print("yvar = " + str(yvar)) 
    print("ycovar = " + str(ycovar))
    print("yintravar = " + str(yintravar))
                     
    return ysim, yvar, ycovar, yintravar
         

class Network_class(nn.Module):
    ''' 
    Define the network used
    '''
    def __init__(self, args):
        super(Network_class, self).__init__()
        self.fc1 = nn.Linear(args.nneurons, 10, bias = True)
    
        if args.device >= 0 and torch.cuda.is_available():
            device = torch.device(args.device)
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        self = self.to(device)
    
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr_classifier, betas=(0.9, 0.999))
    
    def forward(self, x):
        '''
        Simple linear layer that takes as input the features exctracted by the network and output the classes of the input
        '''
        y = self.fc1(x)
        return y 
                 
        
def train_classifier(net, net_class, train_loader_classifier, input_layer = 0):
    '''
    Train the final linear classifier
    1. Collect data from the dataset
    2. Perform a forward pass through the pre-train network which weights have been frozen
    3. Send the final layer of the pre-trained network to the linear classifier
    4. Compute the loss and compute the gradient of only the weights of the linear classifier
    5. Only update the weight of the classifier
    '''
    criterion = nn.CrossEntropyLoss()

    net.eval()
    net.zero_grad()
    
    error, loss_tot = 0, 0
    
    for batch_idx, (data, target) in enumerate(train_loader_classifier):
        net_class.optimizer.zero_grad()
        data, target = data.to(net.device), target.to(net.device)
        
        with torch.no_grad():
            states  = net.forward_simple(data) #simple forward pass with the pre-trained network
        
        output = states[input_layer]
        y = net_class(output) 

        loss = criterion(y, torch.argmax(target, dim=1))
        loss.backward()
        loss_tot += loss.item()
        net_class.optimizer.step()
        error += (torch.argmax(y, dim =1) != torch.argmax(target, dim =1)).sum()
        
    return (error/len(train_loader_classifier.dataset))*100, loss_tot/len(train_loader_classifier.dataset)


def test_classifier(net, net_class, test_loader, input_layer = 0):
    '''
    Test the whole architecture: pre-trained feature extractor + linear classifier
    1. Collect data from the dataset
    2. Perform a forward pass through the pre-train network which weights have been frozen
    3. Send the final layer of the pre-trained network to the linear classifier and compute the loss & prediction
    '''
    criterion = nn.CrossEntropyLoss()
    net.eval()
    net.zero_grad()
    
    error, loss_tot = 0, 0
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(net.device), target.to(net.device)
        
        with torch.no_grad():
            states  = net.forward_simple(data) #simple forward pass with the pre-trained network

        output = states[input_layer]
        
        y = net_class(output) #forward pass classifier

        loss = criterion(y, torch.argmax(target, dim=1))
        loss_tot += loss.item()
        error += (torch.argmax(y, dim =1) != torch.argmax(target, dim =1)).sum()
        
    return (error/len(test_loader.dataset))*100, loss_tot/len(test_loader.dataset)


def training_classifier_loop(args, net, train_loader_classifier, test_loader, input_layer = 0):
    '''
    Function used to instantiate a linear probe on each layer, train it and store the corresponding training curves
    '''
    net_class = Network_class(args)
    train_loss, train_err = [], []
    test_loss, test_err = [], []

    for k in range(args.epochs_classifier):
        error, loss = train_classifier(net, net_class, train_loader_classifier, input_layer = input_layer)
        train_loss.append(loss)
        train_err.append(error.item())
        
        error, loss = test_classifier(net, net_class, test_loader, input_layer = input_layer)
        test_loss.append(loss)
        test_err.append(error.item())

    return train_err, test_err, train_loss, test_loss


def training_classifiers(BASE_PATH, args, net, epoch, train_loader_classifier, test_loader, layer):
    '''
    Train a linear classifier on top of each layer - store the corresponding training curves
    '''
    net.eval()
    
    train_err, test_err, train_loss, test_loss = training_classifier_loop(args, net, train_loader_classifier, test_loader, input_layer = layer)
    name_dataframe = "linear_classifier_layer#" + str(layer) + "_epoch#" + str(epoch) + ".csv"
    save_Dataframe_classifier(BASE_PATH, [train_loss, test_loss, train_err, test_err], dataframe_to_init = name_dataframe)
        
    return 0