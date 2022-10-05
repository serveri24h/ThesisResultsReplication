import numpy as np
import pandas as pd
import torch
import os
from matplotlib import pyplot as plt

from models import GregoryNet, ModifiedNet

import constants as const
import helpers as h


class GP_GRID_SEARCH():
    def __init__(self,params,alpha=2, scale=0.2 ,sigma2=1,prior=1.5, acquasition_rate=2,normalized=True,prints=True):
        self.prints=prints
        self.params = params
        self.dims = len(params)
        self.l = np.prod([len(p) for p in params])

        self.a = alpha
        self.k = scale
        self.s2 = sigma2
        self.aq_rate = acquasition_rate

        self.Xp, self.mapping = self.flatten_data(X0 = [], mapping={})
        self.normalized = normalized
        if self.normalized: 
            self.Xp_norm = self.normalize_Xp()

        if self.prints: print("Initiating the grid for acquasition function...")
        self.initiate_grid(prior)

        self.DATA_X = np.array([])
        self.DATA_y = np.array([])
        self.DATA_epochs = np.array([])
    
    def flatten_data(self, X0, mapping, parents=[], d=0):
        if d == self.dims-1:
            for i in range(len(self.params[d])):
                new_x = parents+[self.params[d][i]] 
                mapping[len(X0)] = new_x
                X0.append( new_x )
        else:
            for i in range( len(self.params[d]) ):
                new_parents = parents.copy()
                new_parents.append(self.params[d][i])
                self.flatten_data(X0, mapping, new_parents,d+1)
        return np.array(X0), mapping
    
    def normalize_Xp(self):
        Xp_norm = np.array(self.Xp,copy=True)
        for d in range(self.dims):
            Xp_norm[:,d] = (self.Xp[:,d]-self.params[d][0])/(self.params[d][-1]-self.params[d][0])
        return Xp_norm

    def get_Xp(self):
        if self.normalized:
            return self.Xp_norm
        else:
            return self.Xp
    
    def normalize_X(self):
        X_norm = np.array(self.DATA_X, copy=True)
        for d in range(self.dims):
            X_norm[:,d] = (X_norm[:,d]-self.params[d][0])/(self.params[d][-1]-self.params[d][0])
        return X_norm

    def get_X(self):
        if self.normalized:
            return self.normalize_X()
        else:
            return self.DATA_X

    def create_se_kernel(self, X1, X2):
        """ returns the NxM kernel matrix between the two sets of input X1 and X2 """

        N = len(X1)
        M = len(X2)
        
        K = np.array([[ np.sum( (X1[n]-X2[m])**2 ) for m in range(M)] for n in range(N)])
        K = self.a*np.exp( -K/(2*(self.k**2)) )
    
        return K

    def kernel_posterior(self):
        """ returns the posterior distribution of f evaluated at each of the points in Xp conditioned on (X, y)
            using the squared exponential kernel. """
        Xp = self.get_Xp()
        X = self.get_X()
        K_ff_inv = np.linalg.inv(self.create_se_kernel(X,X)+self.s2*np.identity(len(X)) )
        K_fsf = self.create_se_kernel(Xp,X)
        K_fsfs = self.create_se_kernel(Xp,Xp)
        
        mu_f = np.dot( K_fsf, np.dot(K_ff_inv,self.DATA_y[:,None]) ) 
        var_f = K_fsfs-np.dot(K_fsf,np.dot(K_ff_inv, K_fsf.T))
        return mu_f, var_f
    
    def initiate_grid(self,prior=2):
        Xp = self.get_Xp()
        self.mu, self.Sigma = (prior*np.ones(len(self.Xp))[:,None], self.create_se_kernel(Xp,Xp))

    def update_posterior(self):
        if self.prints: print("Computing the posterior kernel...")
        self.mu, self.Sigma = self.kernel_posterior()

    def get_next_x(self):
        x_list = self.DATA_X.tolist()
        grid = self.mu-(self.aq_rate*np.diag(self.Sigma))[:,None]
        for i in range( self.l ):
            if self.mapping[i] in x_list:
                grid[i]=10**9
        self.latest = self.mapping[np.argmin(grid)]

    def add_new_data(self, new_x, new_y,new_n_epochs):
        if len(self.DATA_X) == 0:
            self.DATA_X=np.array([new_x])
        else:
            self.DATA_X = np.vstack([self.DATA_X, new_x])
        self.DATA_y = np.append(self.DATA_y,new_y)
        self.DATA_epochs = np.append(self.DATA_epochs,new_n_epochs)

    def add_new_data_batch(self, X, y, n_epochs):
        if len(self.DATA_X) == 0:
            self.DATA_X=X
            self.DATA_y=y
            self.DATA_epochs = n_epochs
        else:
            self.DATA_X = np.vstack([self.DATA_X, X])
            self.DATA_y = np.append(self.DATA_y,y) 
            self.DATA_epochs = np.append(self.DATA_epochs,n_epochs) 
    
    def add_data_from_file(self,data_path):
        try:
            df = pd.read_csv(data_path)
            data_read = df.to_numpy()
            X = data_read[:,1:-2]
            y = data_read[:,-2]
            n_epochs = data_read[:,-1]
            self.add_new_data_batch(X,y,n_epochs)
        except Exception as e:
            print("Failed to read data with error:\n",e)
    
    def next_parameters(self):
        self.update_posterior()
        self.get_next_x()
        return self.latest
    
    def step(self,y,n_epochs, X=np.array([])):
        if len(X)<1:
            self.add_new_data(self.latest,y,n_epochs)
        else:
            self.add_new_data(X,y,n_epochs)
        return self.next_parameters()

    def clear_data(self):
        self.DATA_X = np.array([])
        self.DATA_y = np.array([])
        self.DATA_epochs = np.array([])

class hyperparameter_optimizer():
    def __init__(self,data,gpgs=None,model="Modified",outdir=None,device=torch.device('cpu'),criterion=torch.nn.MSELoss(),prints=True):
        self.DATA = data
        self.GPGS = gpgs
        self.DEVICE = device
        self.criterion = criterion
        self.model=model
        if outdir:
            self.outdir = outdir
        else:
            self.outdir = os.getcwd()
        self.prints=prints

    def compute_test_error(self,NET):
        # SET EVALUATION MODE
        NET.eval()
        
        # PREPARE TEST DATA
        X = self.DATA.validation_batch()
        
        # PREDICTION
        with torch.no_grad():
            pred = NET.forward(X[0:4])
        
        # ERROR
        true = X[4]
        test_error = self.criterion(pred,true).item()
        
        # SET TRAINING MODE
        NET.train()
        
        return test_error

    def training_loop(self,NET,optimizer,epochs=1):
        if self.prints: print("Starting a training loop for {} epoch(s)".format(epochs))
        # SETUP
        Es_train = []
        Es_test = []
        
        # COMPUTE ERROR BEFORE ANY TRAINING
        test_error = self.compute_test_error(NET)
        Es_test.append(test_error)
        if self.prints: print("Test error for before training: {:2f}".format(test_error))
        
        # TRAINING LOOP
        for epc in range(epochs):
            if self.prints: print("EPOCH:", epc+1)
                
            # START EPOCH
            self.DATA.start_epoch()
            NET.train()
            while self.DATA.epoch_status():
                # RESET GRADIENT
                NET.zero_grad()
                
                # PREPARE TRAINING DATA
                X = self.DATA.training_batch()
                
                # FORWARD
                pred = NET.forward(X[0:4])

                # ERROR/LOSS
                true = X[4]
                loss = self.criterion(pred,true)
                if torch.isnan(loss) or torch.isinf(loss):
                    return [3],[3] # Escape from failed model
                if self.prints: print("PROCESS:  {0:.4f} %  ------  ERROR: {1:.2f}".format( self.DATA.get_progress(),loss.item() ) )
                Es_train.append(loss.item())
                
                # BACKPROPAGATION
                loss.backward()
                optimizer.step()
            
            # EVALUATION
            test_error = self.compute_test_error(NET)
            Es_test.append(test_error)
            if self.prints: print("Test error for EPOCH: {0:} = {1:2f}".format(epc+1,test_error))
                
        return Es_train,Es_test

    def evaluate_parameters(self,parameters,epochs):
        # MODEL
        if parameters['model'] == 'Modified':
            NET = ModifiedNet(parameters,self.DEVICE)
        else:
            print("YOU SHOULDN'T BE HERE")
            NET = GregoryNet(parameters).to(self.DEVICE)
        
        # LEARNING PARAMETERS
        self.DATA.update_batch_size(parameters['batch_size'])
        lr = 10**(-float(parameters['learning_rate']))
        mf = parameters['momentum_factor']
        optimizer = torch.optim.SGD(NET.parameters(), lr=lr, momentum=mf)
        Es_train,Es_test = self.training_loop(NET,optimizer,epochs=epochs)
        min_epoch = np.argmin(Es_test)
        return Es_test[min_epoch], min_epoch

    def get_params_dict(self,values,mf=0.9,hs=None,hl=1,ks=4):
        lr = values[0]
        bs = values[1]
        hs = values[2]
        if self.model == 'Gregory':
            ks = values[2]

        # PARAMETERS DETERMINED BY THE DATA
        d_max,d_alpha,p_max,p_alpha,n_family = self.DATA.get_dimensions()
        
        # OTHER PRE-DETERMINED VARIABLES
        N_conv_channels = const.N_CONV_CHANNELS
        output_size = const.OUTPUT_SIZE
        fp_size = const.FP_SIZE
        embed_size_protein = const.EMBED_SIZE_PROTEIN
        embed_size_drug = const.EMBED_SIZE_DRUG
        
        parameters = {'model':self.model,
                    'learning_rate':lr,
                    'momentum_factor':mf,
                    'batch_size': int(bs), 
                    'p_max':p_max,
                    'd_max':d_max, 
                    'p_alpha':p_alpha,
                    'd_alpha':d_alpha,
                    'n_family':n_family,
                    'h_size_p':int(hs),
                    'h_size_d':int(hs),
                    'h_layers':hl,
                    'n_channel':N_conv_channels,
                    'output_size':output_size,
                    'fp_size':fp_size,
                    'kernel_size':ks,
                    'embed_size_p':embed_size_protein,
                    'embed_size_d':embed_size_drug}
        return parameters

    ################################
    ############ TOOOLS ############
    ################################

    def plot_grid(self,new_x,scale=1.8,name='GPGS.png'):
        X = self.GPGS.DATA_X
        y = self.GPGS.DATA_y
        plot_color = [['red','blue'][int(bool_y)] for bool_y in y<1.5]
        plot_size = (np.exp(-y/scale)*30)**2
        self.ax.clear()
        self.ax = plt.axes(projection='3d')

        # Residuals
        for i in range(len(X)):self.ax.plot([X[i,0]]*2, [X[i,1]]*2, [X[i,2],0],c='grey')
        self.ax.plot([new_x[0]]*2, [new_x[1]]*2, [new_x[2],0],c='green')

        # 3d-Points
        self.ax.scatter(X[:,0], X[:,1], X[:,2],s=plot_size,c=plot_color)
        self.ax.scatter(new_x[0],new_x[1],new_x[2],color='green')

        # Labels
        self.ax.set_xlabel('lr', fontsize=20)
        self.ax.set_ylabel('bs', fontsize=20)
        self.ax.set_zlabel('hs', fontsize=20)
        plt.savefig(self.outdir+name)
    
    def save_data(self,file_name='results.csv'):
        X = self.GPGS.DATA_X
        y = self.GPGS.DATA_y[:,None]
        epochs = self.GPGS.DATA_epochs[:,None]
        save_data = np.concatenate((X,y,epochs),axis=1)
        save_data = pd.DataFrame(save_data)
        save_data.to_csv(self.outdir+file_name)

    def load_data(self, file_name):
        self.GPGS.add_data_from_file(file_name)
    
    def predict_minimum(self):
        try:
            next_pred = self.GPGS.latest
        except:
            next_pred = self.GPGS.next_parameters()
        idx_min = np.argmin(self.GPGS.DATA_y)
        empirical_minimum = self.GPGS.DATA_X[idx_min].tolist()
        epochs = int(self.GPGS.DATA_epochs[idx_min])
        posterior_minimum = self.GPGS.Xp[np.argmin(self.GPGS.mu)].tolist()
        return empirical_minimum, posterior_minimum, next_pred, epochs


    ################################
    ########## MAIN ################
    ################################
    
    def run(self,values,N_search=5,epochs=2,reset_previous=True):
        if reset_previous:
            self.GPGS.clear_data()
        
        # INITIATE PLOT AX
        self.fig = plt.figure(figsize=(12,12))
        self.ax = plt.axes(projection='3d')
        next_values=values

        # Search with gaussian process
        for n in range(N_search):
            if self.prints: print("Evaluating parameters Learning rate = {}, Batch size = {}, Hidden size = {}".format(next_values[0],next_values[1],next_values[2]))
            if len(self.GPGS.DATA_X)<1:
                # FIRST SEARCH
                lr_0 = values[0]
                bs_0 = values[1]
                hs_0 = values[2]
                values = [lr_0,bs_0,hs_0]
                params0 = self.get_params_dict(values,hl=const.HIDDEN_LAYERS)
                y,min_epoch = self.evaluate_parameters(params0, epochs)
                next_values = self.GPGS.step(y,min_epoch,np.array(values))
            else:
                params = self.get_params_dict(next_values,hl=const.HIDDEN_LAYERS)
                y,min_epoch = self.evaluate_parameters(params,epochs)
                next_values = self.GPGS.step(y,min_epoch,next_values)
            self.plot_grid(next_values)
            self.save_data()
        
    def init_model(self, values, model='Modified'):
        # MODEL
        params = self.get_params_dict(values,hl=const.HIDDEN_LAYERS)

        if model == 'Modified':
            NET = ModifiedNet(params,self.DEVICE).to(self.DEVICE)
        else:
            NET = GregoryNet(params).to(self.DEVICE)

        self.DATA.update_batch_size(params['batch_size'])
        lr = 10**(-float(params['learning_rate']))
        mf = params['momentum_factor']
        optimizer = torch.optim.SGD(NET.parameters(), lr=lr, momentum=mf)

        return NET,optimizer
    
    def train(self,NET,optimizer,epochs):
        # LEARNING PARAMETERS
        return self.training_loop(NET,optimizer,epochs=epochs)

if __name__=="__main__":
    print("This is training tools file")