import numpy as np
import pandas as pd
import torch
import os
from matplotlib import pyplot as plt

from models import GregoryNet, ModifiedNet

import training_tools.training_utilities as tutils
import constants as const
import helpers as h
from context import Context as ctx

def train_model(NET,DATA,CTX):
    # LEARNING PARAMETERS
    optimizer = torch.optim.SGD(NET.parameters(), lr=10**(-float(CTX.free_parameters['learning_rate'])), momentum=CTX.free_parameters['momentum_factor'])
    return tutils.training_loop(NET,DATA,optimizer,CTX,tutils.stop_training_standard)

def evaluate_values(values,DATA,CTX):
    # Initiate Deep Learing model
    CTX.update_model_parameters(values)
    NET = ModifiedNet(CTX).to(CTX.device)
    Es_train,Es_test = train_model(NET,DATA,CTX)
    min_epoch = np.argmin(Es_test)
    return min(Es_test[min_epoch],3), min_epoch

class GP_GRID_SEARCH():
    def __init__(self,CTX,outdir=const.RESULTS_DIR):
        self.prints=CTX.prints
        self.params = CTX.gpgs_parameters['grid']
        self.dims = len(self.params)
        self.l = np.prod([len(p) for p in self.params])

        self.a = CTX.gpgs_parameters['alpha']
        self.k = CTX.gpgs_parameters['scale']
        self.s2 = CTX.gpgs_parameters['sigma2']
        self.aq_rate = CTX.gpgs_parameters['acq_rate']
        self.prio = CTX.gpgs_parameters['prior']

        self.Xp, self.mapping = self.flatten_data(X0 = [], mapping={})
        self.normalized = CTX.gpgs_parameters['normalized']
        if self.normalized: 
            self.Xp_norm = self.normalize_Xp()

        if self.prints: print("Initiating the grid for acquasition function...")
        self.initiate_grid(self.prio)

        self.DATA_X = np.array([])
        self.DATA_y = np.array([])
        self.DATA_epochs = np.array([])
    
        try:
            self.outdir = h.folder_name(outdir)
        except:
            self.outdir = os.getcwd()
    
    # def flatten_data(self, X0, mapping, parents=[], d=0):
    #     if d == self.dims-1:
    #         for i in range(len(self.params[d])):
    #             new_x = parents+[self.params[d][i]] 
    #             mapping[len(X0)] = new_x
    #             X0.append( new_x )
    #     else:
    #         for i in range( len(self.params[d]) ):
    #             new_parents = parents.copy()
    #             new_parents.append( self.params[d][i] )
    #             self.flatten_data( X0, mapping, new_parents,d+1 )
        
    #     for x0 in X0:
    #         print(x0)
    #     print("\n")

    #     return np.array(X0), mapping

    def flatten_data(self, X0, mapping, parents=[], d=1):
        if d == self.dims:
            for i in range(len(self.params[self.dims-d])):
                new_x = [self.params[self.dims-d][i]]+parents
                mapping[len(X0)] = new_x
                X0.append( new_x )
        else:
            for i in range( len(self.params[self.dims-d]) ):
                new_parents = parents.copy()
                self.flatten_data( X0, mapping,[self.params[self.dims-d][i]] + new_parents,d+1 )

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
    
    def normalize_X(self, X=None):
        if X is None:
            X_norm = np.array(self.DATA_X, copy=True)
        else:
            X_norm = np.array(X)
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

    def kernel_posterior(self,Xp=None):
        """ returns the posterior distribution of f evaluated at each of the points in Xp conditioned on (X, y)
            using the squared exponential kernel. """
        if Xp is None:
            Xp = self.get_Xp()
        else:
            Xp = self.normalize_X(Xp)
        X = self.get_X()
        K_ff_inv = np.linalg.inv(self.create_se_kernel(X,X)+self.s2*np.identity(len(X)) )
        K_fsf = self.create_se_kernel(Xp,X)
        K_fsfs = self.create_se_kernel(Xp,Xp)
        
        mu_f = np.dot( K_fsf, np.dot(K_ff_inv,(self.DATA_y-self.prio)[:,None]) )+self.prio 
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
        return self.mapping[np.argmin(grid)]

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
        return self.get_next_x()

    def clear_data(self):
        self.DATA_X = np.array([])
        self.DATA_y = np.array([])
        self.DATA_epochs = np.array([])

    #################################################   
    #######        MAIN FUNCTIONALITITES       ######
    #################################################

    def predict_minimum(self):
        next_pred = self.next_parameters()
        idx_min = np.argmin(self.DATA_y)
        empirical_minimum = self.DATA_X[idx_min].tolist()
        epochs = int(self.DATA_epochs[idx_min])
        posterior_minimum = self.Xp[np.argmin(self.mu)].tolist()
        return empirical_minimum, posterior_minimum, next_pred, epochs

    def get_posterior_minimum(self):
        y_min = np.min(self.mu)
        x_min = self.Xp[np.argmin(self.mu)]
        return x_min,y_min
    
    def run(self,DATA,CTX):
        # FIRST VALUE
        values = [100,3,0.5]
        if CTX.load:
            try:
                self.add_data_from_file(const.RESULTS_DIR+"results.csv")
                _1,_2,values,_3=self.predict_minimum()
            except:
                print("Unable to load data. Starting with initial guess.")

        # Search with gaussian process
        for n in range(CTX.gpgs_parameters['n_search']):
            if CTX.prints: print(f"Evaluating parameters: ",CTX.gpgs_parameters['param_names'],"With values: ",values)
            y,min_epoch = evaluate_values(values,DATA,CTX)
            self.add_new_data(values,y,min_epoch)
            if CTX.save: 
                self.save_data()
            if n<CTX.gpgs_parameters['n_search']-1:
                values = self.next_parameters()


    def plot_2d_grid(self,CTX,name='GPGS2D.png',rotation=[35,35]):
        fig = plt.figure(figsize=(10,10))#figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1,1,1,projection='3d')
        
        xx,yy = np.meshgrid(CTX.gpgs_parameters['grid'][0],CTX.gpgs_parameters['grid'][1])

        std = np.reshape(np.sqrt(np.diag(self.Sigma)),(-1,CTX.gpgs_parameters['precision']))
        m = np.reshape(self.mu, (-1,CTX.gpgs_parameters['precision']))

        upper = m+std
        lower = m-std

        x_min,y_min = self.get_posterior_minimum()

        #ax.plot_surface(xx,yy,upper,color='grey',alpha=0.6)
        #ax.plot_surface(xx,yy,m,color='red',alpha=0.1)
        ax.plot_wireframe(xx,yy,m,color='red',alpha=0.5)
        #ax.plot_surface(xx,yy,lower,color='grey',alpha=0.6)
        ax.scatter(self.DATA_X[:,0],self.DATA_X[:,1],self.DATA_y,linewidths=5,color='black')
        #ax.scatter(x_min[1],x_min[0],y_min,linewidths=10,color='green')

        # RESIDUALS
        for i,m_i in enumerate(self.kernel_posterior(Xp=self.DATA_X)[0]):
            ax.plot([self.DATA_X[i,0]]*2,[self.DATA_X[i,1]]*2,[m_i,self.DATA_y[i]],color='grey')

        ax.scatter(x_min[0],x_min[1],y_min,linewidths=5,s=200,color='green',marker="*")
        ax.set_xlabel(CTX.gpgs_parameters['param_labels'][0],fontsize=18)
        ax.set_ylabel(CTX.gpgs_parameters['param_labels'][1],fontsize=18)
        ax.view_init(rotation[0],rotation[1])
        #plt.savefig(self.outdir+name)
    
    def plot_3d_grid(self,CTX,new_x,scale=1.8,name='GPGS3D.png'):
        X = self.DATA_X
        y = self.DATA_y

        fig = plt.figure(figsize=(10,10))#figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1,1,1,projection='3d')

        plot_color = [['red','blue'][int(bool_y)] for bool_y in y<1.5]
        plot_size = (np.exp(-y/scale)*30)**2
        ax = plt.axes(projection='3d')

        # Residuals
        for i in range(len(X)):
            ax.plot([X[i,0]]*2, [X[i,1]]*2, [X[i,2],0],c='grey')
        ax.plot([new_x[0]]*2, [new_x[1]]*2, [new_x[2],0],c='green')

        # 3d-Points
        ax.scatter(X[:,0], X[:,1], X[:,2],s=plot_size,c=plot_color)
        ax.scatter(new_x[0],new_x[1],new_x[2],color='green')

        # Labels
        ax.set_xlabel('bs', fontsize=20)
        ax.set_ylabel('lr', fontsize=20)
        ax.set_zlabel('mf', fontsize=20)
        plt.savefig(self.outdir+name)
    
    def save_data(self,file_name='results.csv'):
        X = self.DATA_X
        y = self.DATA_y[:,None]
        epochs = self.DATA_epochs[:,None]
        save_data = np.concatenate((X,y,epochs),axis=1)
        save_data = pd.DataFrame(save_data)
        save_data.to_csv(self.outdir+file_name)

if __name__=="__main__":
    print("This is training tools file")