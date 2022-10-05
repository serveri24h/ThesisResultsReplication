import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as const

def load_model(file_name = "model.pickle"):
    try:
        model_dict = torch.load(file_name)
        CTX = model_dict['context']
        W_dict = model_dict['model']
        NET = ModifiedNet(CTX).to(CTX.device)
        NET.load_state_dict(W_dict)
        return NET, CTX
    except Exception as e:
        print(f"Failed to load model with following error:\n{e}")
        return None, None

def save_model(NET, CTX, file_name = "model.pickle"):
    try:
        model_dict = {'model': NET.state_dict(), 'context': CTX}
        torch.save(model_dict,file_name)
    except Exception as e:
        print(f"Failed to save thw model with following error:\n{e}")

##############################
## The Top Performing Model ##
##############################

class ModifiedNet(nn.Module):
    #NNI stand for neural network for the Interaction
    def __init__(self,CTX):
        super(ModifiedNet, self).__init__()
        self.device = CTX.device
        self.nnp = RCN(CTX.data_parameters['p_alpha'],CTX.free_parameters['n_channel'],
                        CTX.free_parameters['h_layers'],CTX.free_parameters['h_size_p'],CTX.free_parameters['output_size'],CTX.free_parameters['embed_size_p'],self.device).to(self.device )
        self.nnd = RCN(CTX.data_parameters['d_alpha'],CTX.free_parameters['n_channel'],
                        CTX.free_parameters['h_layers'],CTX.free_parameters['h_size_d'],CTX.free_parameters['output_size'],CTX.free_parameters['embed_size_d'],self.device).to(self.device )
        self.fc = nn.Linear(CTX.free_parameters['output_size']*2+CTX.free_parameters['fp_size']+CTX.data_parameters['n_family'],32).to(self.device )
        self.dropout = nn.Dropout(p=0.4) 
        self.output = nn.Linear(32,1).to(self.device)

    def forward(self, X):
        x = torch.cat((self.nnd(X[0]),self.nnp(X[1]),X[2],X[3]),axis=1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class RCN(nn.Module):
    #RCNN stands for recurrent convolutional Neural network
    def __init__(self, alphabeth_size, N_conv_channels, hidden_layers, hidden_size, output_size, embed_size, device):
        super(RCN, self).__init__()

        # DEVICE
        self.device=device
        
        # Local Variables
        self.hidden_size = int(hidden_size)
        self.hidden_layers = int(hidden_layers)
        self.conv_channels = int(N_conv_channels)
        
        #Recurrent unit
        self.embedding = nn.Embedding(alphabeth_size+1,int(embed_size))
        self.rnn = nn.GRU(int(embed_size), self.hidden_size, num_layers=self.hidden_layers, batch_first=True)
        
        #Convolutional layer
        self.conv = nn.Conv1d(1, self.conv_channels, 4, 2)
        self.dropout = nn.Dropout(p=0.5) 
        self.fc = nn.Linear(self.conv_channels*int(self.hidden_size/2-4),output_size)
    
    def pad_sequences(self,x):
        x_lengths = [list(x_i.size())[0] for x_i in x]
        max_seq_length = max(x_lengths)
        x_0 = torch.ones(len(x),max_seq_length,dtype=torch.long).to(self.device)*const.PADDING_VALUE
        for i in range(len(x)):
            x_0[i,0:list(x[i].size())[0]] = x[i]
        return x_0, x_lengths

    def forward(self, x):
        x,x_lengths = self.pad_sequences(x)
        self.h0=torch.zeros(self.hidden_layers,len(x),self.hidden_size).to(self.device)
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x,x_lengths,batch_first=True,enforce_sorted=False)
        _,x = self.rnn(x,self.h0)
        x = x[-1].view(-1,1,self.hidden_size)
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, 4,1)
        x = self.dropout(x)
        x = x.view(-1,self.conv_channels*int(self.hidden_size/2-4))
        x = self.fc(x)
        return x

#####################
## OLD RCNN METHOD ##
#####################

class RCNN(nn.Module):
    #RCNN stands for recurrent convolutional Neural network
    def __init__(self, alphabeth_size, N_conv_channels, hidden_layers, hidden_size, output_size, device):
        super(RCNN, self).__init__()

        # DEVICE
        self.device=device
        
        # Local Variables
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.conv_channels = N_conv_channels
        self.h0=None
        
        #Recurrent unit
        # LSTM inputs (input_size,hidden_size,num_layers)
        self.rnn = nn.GRU(alphabeth_size,hidden_size, num_layers=hidden_layers, batch_first=True)
        
        #Convolutional layer
        self.conv = nn.Conv1d(hidden_layers, N_conv_channels, 4, 2)
        self.fc = nn.Linear(N_conv_channels*int(hidden_size/2-4),output_size)
    
    def h0_exists(self):
        if self.h0 != None: return True
        return False

    def forward(self, x, read_shape=True):
        if read_shape or not self.h0_exists:
            self.h0 = torch.zeros(self.hidden_layers,x.shape[0],self.hidden_size).to(self.device)
        _,x = self.rnn(x,self.h0) 
        x = x.view(-1,self.hidden_layers,self.hidden_size)
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, 4,1)
        x = F.dropout(x,0.5)
        x = x.view(-1,self.conv_channels*int(self.hidden_size/2-4))
        x = self.fc(x)
        return x

#####################################
## Replicate of the Gregoty method ##
#####################################

class GregoryNet(nn.Module):
    #NNI stand for neural network for the Interaction
    def __init__(self,K):
        super(GregoryNet, self).__init__()
        self.nnp = CNN64(K['p_max'],K['p_alpha'],K['n_channel'],K['output_size'],K['kernel_size'])
        self.nnd = CNN64(K['d_max'],K['d_alpha'],K['n_channel'],K['output_size'],K['kernel_size'])
        self.fc = nn.Linear(K['output_size']*2+K['fp_size']+K['n_family'],32)
        self.output = nn.Linear(32,1)

    def forward(self, X):
        x = torch.cat((self.nnd(X[0]),self.nnp(X[1]),X[2],X[3]),axis=1)
        x = F.dropout(self.fc(x),0.4)
        x = self.output(x)
        return x
    
class CNN64(nn.Module):
    #CNN64 stand for convolutional neural network with 64 channels
    def __init__(self,max_seq_length,alphabet_size,N_channels,output_size,kernel=2):
        super(CNN64, self).__init__()
        
        # Local Variables
        self.N_channels = N_channels
        self.kernel = kernel
        self.N_fc_inputs = N_channels*int((max_seq_length-1)/self.kernel)*int((alphabet_size-1)/self.kernel)

        #Convolutional layer
        # Input channels = 1, Output channels = 64, (Assumed) kernel Size = 2
        self.conv = nn.Conv2d(1, N_channels, self.kernel)
        
        #FC-layer
        # Number of inputs calculated in local variables
        # Output size = 256
        self.fc = nn.Linear(self.N_fc_inputs, output_size)

    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.conv(x)) # Convolutional layer with relu activation
        x = F.max_pool2d(x, (self.kernel, self.kernel)) #  Maxpool with (2,2) kernel
        x = F.dropout(x,0.5)
        x = x.view(-1,self.N_fc_inputs)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    print("This is the model file")
