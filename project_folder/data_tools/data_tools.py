## PACKAGES ##
import warnings
import pandas as pd
import numpy as np
import torch
from math import ceil

# LOCAL MODULES
import data_tools.data_utilities as dutils 
import constants as const
import helpers as h

########################################
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ #
# \/\/\/\/ BATCH TOOL CLASS \/\/\/\/\/ #
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ #
########################################

class BATCH_TOOL():
    def __init__(self, data_size, batch_size=50):
        self.bs = batch_size
        self.ds = data_size
        self.epoch = False
        self.steps = ceil(data_size/batch_size)
        self.p = 0

    def step(self):
        ret = (self.p,self.bs)
        self.p += 1
        if self.p == self.steps:
            self.epoch = False
        return ret
        
    def epoch_status(self):
        return self.epoch

    def start_epoch(self):
        self.epoch = True
        self.p = 0
    
    def update_batch_size(self,new_batch_size):
        self.bs = new_batch_size
        self.steps = ceil(self.ds/self.bs)
    
    def get_progress(self):
        return self.p/self.steps
    
    def get_n_batches(self):
        return self.steps
    
    def get_batch_size(self):
        return self.bs

########################################
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ #
# \/\/\/\/\ DATA TOOL CLASS \/\/\/\/\/ #
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\ #
########################################

class DTI_DATA_TOOL():

    ########################################################
    ########            INIT CLASS                    ######
    ######## ---------------------------------------- ######
    ######## This Data-Tool class requires no inputs. ######
    ######## However, it is recommended to configure  ######
    ######## at least the folders at "src_folders".   ######
    ########################################################

    def __init__(self,standard_types=['KD','KI','IC50','EC50'], device=torch.device("cpu"), validation_frac=0.0025, prints=True, warnings=True,
                    src_data=const.DATA_DIR, src_intermediate = const.INTERMEDIATE_DIR,
                    src_train=const.TRAINING_FILE, src_test=const.TEST_FILE, src_kinase=const.KINASE_FILE, src_fasta=const.FASTA_FILE):
        # SET SOURCE FOLDER FOR THE DATA
        
        # SET STANDARD TYPES
        self.standard_types = standard_types
        
        # FRACTION OF DATA THAT IS USED FOR TESTING
        self.validation_frac = validation_frac
        self.device=device
        self.prints=prints
        self.warnings=warnings
        self.ENV=dutils.PROPER_ENV

        # INITIATE VARIABLES FOR RAW DATA 
        self.keys_train = None
        self.keys_test = None
        
        # SET SOURCE FOLDERS FOR THE DATA
        self.src_data = h.folder_name(src_data)
        self.src_intermediate = h.folder_name(src_intermediate)

        # SET SOURCE FILES FOR THE DATA
        self.src_train = src_train
        self.src_test = src_test
        self.src_kinase = src_kinase
        self.src_fasta = src_fasta

    ########################################################
    ########  USER-LEVEL DATA PREPARATION FUNCTIONS   ######
    ######## ---------------------------------------- ######
    ########  These functions are ment to be called   ######
    ######## by the main function or other program.   ######
    ########################################################

    def prepare_full(self, index_range=None):
        if self.ENV:
            self.prepare_intermediates(index_range=index_range)
            self.prepare_from_intermediates()
        else:
            raise(Exception("Not possible to execute full data preparation with the current enviroment."))
    
    def prepare_and_save(self, index_range=None, src_smiles=const.ID_SMILES_FILE, src_target=const.ID_AMINO_FILE, src_fps=const.ID_FPS_FILE, 
                            src_int_train=const.INTERMEDIATE_TRAIN_FILE,src_int_test1=const.INTERMEDIATE_TEST_FILE1, src_int_test2=const.INTERMEDIATE_TEST_FILE2):
        if self.ENV:                   
            self.prepare_intermediates(index_range=index_range)
            self.save_intermediates(src_smiles=src_smiles, src_target=src_target, src_fps=src_fps, src_int_train=src_int_train, src_int_test1=src_int_test1, src_int_test2=src_int_test2)
        else:
            raise(Exception("Not possible to prepare intermediate data with the current enviroment."))
    
    def load_and_prepare(self, src_smiles=const.ID_SMILES_FILE, src_target=const.ID_AMINO_FILE, src_fps=const.ID_FPS_FILE, 
                            src_int_train=const.INTERMEDIATE_TRAIN_FILE, src_int_test1=const.INTERMEDIATE_TEST_FILE1, src_int_test2=const.INTERMEDIATE_TEST_FILE2):
        if self.load_intermediates(src_smiles=src_smiles,src_target=src_target,src_fps=src_fps,src_int_train=src_int_train,src_int_test1=src_int_test1, src_int_test2=src_int_test2):
            self.prepare_from_intermediates()
        else:
            raise(Exception("Failed to load intermediate files."))

    ########################################################
    ########    DATA-PREPARATION-LEVEL FUNCTIONS      ######
    ######## ---------------------------------------- ######
    ######## These top-level functions are the core   ######
    ########   of the data preparation pipeline.      ######
    ########################################################

    def read_training_data(self,index_range=None):
        self.keys_train = dutils.read_training_data_from_file(src_dti=self.src_data+self.src_train,index_range=index_range)

    def prepare_intermediates(self, index_range=None):
        # READ TRAINING DATA
        try:
            self.read_training_data(index_range=index_range)
        except:
            print("ERROR! Failed to read training data.")
            raise Exception("Training data is required!!")

        # GET KINASE INFORMATION
        try:
            self.kinase_info, self.kinase_family = dutils.read_kinase_information_from_file(src_kinase=self.src_data+self.src_kinase)
        except:
            print("ERROR! Kinase information could not be read")
            raise Exception("Kinase information is required!!")

        # FILTER THE TRAINING DATA
        self.keys_train = dutils.filter_training_data(df_dti=self.keys_train,kinase_info=self.kinase_info,standard_types=self.standard_types)

        # READ TEST DATA
        try:
            self.read_test_data()
        except:
            self.keys_test = None
            print("WARNING! Could not read test data. Will continue routine without.")
        
        # LIST OF UNIQUE DRUGS AND PROTEINS PRESENT IN THE DATA
        self.ids_drug, self.ids_target = list(self.keys_train.compound_id.unique()), list(self.keys_train.target_id.unique())

        # FILTER OUT TEST SAMPLES FROM TRAINING DATA TO AVOID TRAINING BIAS
        if self.keys_test is not None:
            self.keys_train = dutils.filter_training_data_with_test_data(self.keys_train, pd.concat( self.keys_test ) )

        # PREPARE SMILES STRINGS
        if self.prints: print("Processing drug compounds...")
        self.DATA_drug = dutils.prepare_smiles(ids_drug=self.ids_drug,prints=self.prints)

        # PREPARE AMINO ACID STRINGS
        if self.prints: print("Processing proteins...")
        self.DATA_target = dutils.get_amino_acids_from_fasta(ids_target=self.ids_target, fasta_file=self.src_data+self.src_fasta, prints=self.prints)

        # PREPARE FINGERPRINTS
        if self.prints: print("Processing fingerprints...")
        self.fingerprints_train = dutils.prepare_finger_prints(self.DATA_drug)

        # PREPARE TEST DATA
        if self.prints: ("Processing test data...")
        self.prepare_test_data()
        
        # REMOVE FAILED KEYS
        self.keys_train = self.keys_train.loc[(self.keys_train['compound_id'].isin(self.DATA_drug))&(self.keys_train['target_id'].isin(self.DATA_target))]
    

    def save_intermediates(self, src_smiles, src_target, src_fps, src_int_train,src_int_test1, src_int_test2 ):
        # SAVE SMILES STRINGS
        try:
            h.save_dict_to_csv(self.DATA_drug,self.src_intermediate+src_smiles)
        except Exception as e:
            print("WARNING! Could not save SMILES strings to file \"{}\" with following error: \n{}".format(src_smiles,e))
        
        # SAVE TARGET SEQUENCES
        try:
            h.save_dict_to_csv(self.DATA_target,self.src_intermediate+src_target)
        except Exception as e:
            print("WARNING! Could not save AMINO ACID sequences to file \"{}\" with following error: \n{}".format(src_target,e))

        # SAVE FINGERPRINTS
        try:
            save_dict = {}
            for i in self.fingerprints_train.keys():
                save_dict[i] = ''.join( [str(k) for k in self.fingerprints_train[i].tolist()  ] )
            h.save_dict_to_csv(save_dict,file_name=self.src_intermediate+src_fps)
        except Exception as e:
            print("WARNING! Could not save FINGERPRINTS to file \"{}\" with following error: \n{}".format(src_fps,e))
        
        # SAVE TRAIN DATA
        try:
            h.save_df_to_csv(df=self.keys_train[['compound_id','target_id','standard_value']], file_name=self.src_intermediate+src_int_train)
        except:
            print("WARNING! Couldn't save train data to file \"{}\" with following error: \n{}".format(src_int_train,e))

        # SAVE TEST DATA
        try:
            self.save_test_data(src_int_test1,src_int_test2)
        except Exception as e:
            print("WARNING! Couldn't save test data with following error: \n{}".format(e))
    
    def load_intermediates(self, src_smiles, src_target, src_fps, src_int_train, src_int_test1, src_int_test2):

        # LOAD KINASE INFO
        try:
            self.kinase_info, self.kinase_family = dutils.read_kinase_information_from_file(src_kinase=self.src_data+self.src_kinase)
        except:
            print("HALT! Kinase information could not be read")
            return False

        # LOAD SMILES STRINGS
        try:
            self.DATA_drug = h.load_from_csv(file_name=self.src_intermediate+src_smiles, data_type='dict')  
        except Exception as e:
            print("HALT! Could not retrieve target sequences from the file {} with error \n{}".format(src_target, e ))
            return False

        # LOAD TARGET SEQUENCES
        try:
            load_dict = h.load_from_csv(file_name=self.src_intermediate+src_target, data_type='dict')    #+self.src_intermediate
            for i,x in load_dict.items():
                load_dict[i] = bytes(x[2:-1], 'utf-8')
            self.DATA_target = load_dict
        except Exception as e:
            print("HALT! Could not retrieve target sequences from the file {} with error \n{}".format(src_target, e ))
            return False
        
        # LOAD FINGERPRINTS
        try:
            load_dict = h.load_from_csv(file_name=self.src_intermediate+src_fps , data_type='dict')
            for i in load_dict.keys():
                load_dict[i] = torch.tensor([float(k) for k in load_dict[i] ])
            self.fingerprints_train = load_dict
        except Exception as e:
            print("HALT! Could not retrieve fingerprints from the file {} with error \n{}".format(src_fps, e ))
            return False
        
        # LOAD TRAIN DATA
        try:
            self.keys_train = h.load_from_csv(file_name=self.src_intermediate+src_int_train)
        except Exception as e:
            print("HALT! Could not retrieve train data from the file {} with error \n{}\n".format(src_fps, e))
            return False
        
        # LOAD TEST DATA
        try:
            self.load_test_data(src_int_test1,src_int_test2)
        except Exception as e:
            print("Warning! Could not retrieve test data from the file {} with error \n{}\n Continuing process without test data.".format(src_fps, e))
        return True

    def prepare_from_intermediates(self):
    
        # TOKENIZE KINASE FAMILIES (each family is represented with unique integer key)
        self.family_token_keys = dutils.tokenize_kinase_families(self.kinase_family, self.keys_train)

        # CLEAN THE RAW TRAINING DATA BY DELETING ALL THE FAILED SAMPLES
        self.keys_train = self.keys_train.loc[(self.keys_train['compound_id'].isin(self.DATA_drug))&(self.keys_train['target_id'].isin(self.DATA_target))]

        # CREATE ONE-HOT-CODED DRUG DATA MATRICES FROM SMILES STRINGS
        if self.prints: print("Preparing Data matrices...")
        self.drug_token_keys, self.drug_matrix_shape = dutils.create_token_keys(self.DATA_drug)
        #self.DATA_drug_matrices = dutils.create_matrix_database(self.DATA_drug,self.drug_token_keys,self.drug_matrix_shape)
        self.DATA_drug_vectors = dutils.create_vector_database(self.DATA_drug,self.drug_token_keys)

        # CREATE ONE-HOT-CODED TARGET DATA MATRICES FROM AMINO ACID STRINGS
        self.target_token_keys, self.target_matrix_shape = dutils.create_token_keys(self.DATA_target)
        #self.DATA_target_matrices = dutils.create_matrix_database(self.DATA_target,self.target_token_keys, self.target_matrix_shape)
        self.DATA_target_vectors = dutils.create_vector_database(self.DATA_target,self.target_token_keys)
        
        # DIVIDE TRAINING DATA INTO TRAINING AND VALIDATION SET
        self.devide_datasets(self.validation_frac)

        # INITIATE BATCH TOOL
        self.BT = BATCH_TOOL(len(self.TRAIN))


    #############################################################
    ############## FUNCTIONS FOR MANAGING TEST DATA ############# 
    #############################################################
    
    def read_test_data(self):
        test_dti1 = dutils.read_test_data_from_file(file_name_data=self.src_data+self.src_test, sheet='Round 1')
        test_dti2 = dutils.read_test_data_from_file(file_name_data=self.src_data+self.src_test, sheet='Round 2')
        self.keys_test = [test_dti1,test_dti2]
    
    def prepare_test_data_file(self,dataset_index):
        df = self.keys_test[dataset_index]
        df['fps'] = dutils.create_finger_prints(df['SMILES'])
        df['AMINOSEQ'] = [None]*len(df)
        amino_dict = dutils.get_amino_acids_from_fasta(df['target_id'].tolist(),fasta_file=self.src_data+self.src_fasta,prints=self.prints)
        for i in range(len(df)):
            df.loc[i,'fps'] = ''.join( [str(k) for k in df.iloc[i]['fps'] ] )
            try:
                df.loc[i,'AMINOSEQ'] = amino_dict[df.loc[i,'target_id']]
            except Exception as e:
                if self.warnings: print("Unable to retrive Amino Acid sequence with uniprot id:",df.loc[i,'target_id'])
        return df
        
    def prepare_test_data(self):
        self.keys_test[0] = self.prepare_test_data_file(0)
        self.keys_test[1] = self.prepare_test_data_file(1)
    
    def save_test_data(self, src_int_test1,src_int_test2):
        h.save_df_to_csv(df=self.keys_test[0][['compound_id','target_id','SMILES','AMINOSEQ','fps','standard_value']], file_name=self.src_intermediate+src_int_test1)
        h.save_df_to_csv(df=self.keys_test[1][['compound_id','target_id','SMILES','AMINOSEQ','fps','standard_value']], file_name=self.src_intermediate+src_int_test2)
    
    def load_test_data(self,src_int_test1,src_int_test2):
        d1 = h.load_from_csv(file_name=self.src_intermediate+src_int_test1)
        d2 = h.load_from_csv(file_name=self.src_intermediate+src_int_test2)
        self.keys_test=[d1,d2]

    #############################################################
    #################     BATCH TOOLS    ########################
    #############################################################

    def prepare_matrix_batch(self,X):
        X_d = torch.stack( list( map(self.DATA_drug_matrices.get, X['compound_id']) ) )
        X_t = torch.stack( list( map(self.DATA_target_matrices.get, X['target_id']) ) )
        
        X_fp = torch.stack( list( map(self.fingerprints_train.get, X['compound_id']) ) )
        X_family = dutils.one_hot_code_family( list( map(self.kinase_family.get, X['target_id']) ), self.family_token_keys )
        
        y = dutils.get_affinity(list(X['standard_value'])) 
        ### HOX! The line below sets the training data into same domain with the test data
        y[y<5]=5

        return X_d.to(self.device), X_t.to(self.device), X_fp.to(self.device), X_family.to(self.device), y.to(self.device)
    
    def prepare_vector_batch(self,X):
        X_d = list( map(self.DATA_drug_vectors.get, X['compound_id']) ) 
        X_t = list( map(self.DATA_target_vectors.get, X['target_id']) ) 
  
        X_fp = torch.stack( list( map(self.fingerprints_train.get, X['compound_id']) ) )
        X_family = dutils.one_hot_code_family( list( map(self.kinase_family.get, X['target_id']) ), self.family_token_keys )
        
        
        y = dutils.get_affinity(list(X['standard_value'])) 
        ### HOX! The line below sets the training data into same domain with the test data
        y[y<5]=5

        return X_d, X_t, X_fp.to(self.device), X_family.to(self.device), y.to(self.device)

    def training_batch(self):
        p,bs = self.BT.step()
        X = self.TRAIN[p*bs:(p+1)*bs] 
        return self.prepare_vector_batch(X)

    def validation_batch(self):
        X = self.VALIDATION
        return self.prepare_vector_batch(X)

    def random_batch(self,batch_size = 50, only_train=False):
        if only_train:
            p = np.random.randint((len(self.TRAIN)-batch_size)/batch_size)
            X = self.TRAIN[p*batch_size:(p+1)*batch_size]
        else:
            p = np.random.randint((len(self.keys_train)-batch_size)/batch_size)
            X = self.keys_train[p*batch_size:(p+1)*batch_size]
        return self.prepare_vector_batch(X)

    def prepare_test_batch(self,dataset_index):
        if self.keys_test is not None and self.keys_train is not None:
            # Select dataset
            DATA = self.keys_test[dataset_index]

            # Affinity
            aff = torch.tensor(DATA['standard_value'])

            # Drug Data
            d_s = DATA['SMILES'].values.tolist()
            d_M,n_success1 = dutils.strings_to_vectors(d_s,self.drug_token_keys,self.drug_matrix_shape,prints=self.prints,warnings=self.warnings)

            # Protein Data
            t_s = DATA['AMINOSEQ']
            t_s_bytes = []
            for s in t_s:
                t_s_bytes.append(bytes(s[2:-1], 'utf-8'))
            t_M,n_success2 = dutils.strings_to_vectors(t_s_bytes,self.target_token_keys,self.target_matrix_shape,prints=self.prints,warnings=self.warnings)

            # Fingerprints
            fps = []
            for s in DATA['fps']:
                fps.append( torch.tensor( [float(x) for x in list(s)] ) )
            fp = torch.stack( fps ) 

            # Kinase Family
            kf = dutils.one_hot_code_family( list(map(self.kinase_family.get, DATA['target_id'])), self.family_token_keys,warnings=False )

            # Remove failed and append
            keep = list(set.intersection(*map(set, [n_success1, n_success2])))
            if len(keep)!=len(DATA):
                if self.warnings: print( "WARNING! Samples removed from test set {}/{}".format(len(keep),len(DATA) ) )
            return [d_M[i] for i in keep],[t_M[i] for i in keep],fp[keep].to(self.device),kf[keep].to(self.device),aff[keep].to(self.device)
        else:
            print("HOX!!! Test data could not be prepared. Please read test data and prepare training data before preparing the test set.")
            return None

    def devide_datasets(self, p=0.01):
        self.TRAIN = self.keys_train.sample(frac=1-p)#,random_state=200)
        self.VALIDATION = self.keys_train.drop(self.TRAIN.index)

    def randomize_dataset(self):
        self.TRAIN = self.TRAIN.sample(frac=1)

    def start_epoch(self):
        self.randomize_dataset()
        self.BT.start_epoch()
    
    def update_batch_size(self,new_batch_size):
        self.BT.update_batch_size(new_batch_size)

    #############################################################
    ###################### ADDITIONAL INFO ######################
    #############################################################

    def print_info(self):
        N_DTI = len(self.keys_train)
        sample_dti = self.keys_train.iloc[0]
        
        # Drug characteristics
        sample_drug = sample_dti['compound_id']
        N_d = len(self.DATA_drug_matrices)
        shape_d = list( self.DATA_drug_matrices[sample_drug].shape )

        # Protein characteristics
        sample_prot = sample_dti['target_id']
        N_p = len(self.DATA_target_matrices)
        shape_p = list( self.DATA_target_matrices[sample_prot].shape )
        N_families = len(self.family_token_keys)

        # PRINT INFO
        print("The Dataset holds affinity for {} drug-kinase pairs\n".format(N_DTI))
        print(" Number of drugs in the dataset = {} \n shape of a drug input matrix [max length, alphabet size] = {}\n".format(N_d,shape_d))
        print(" Number of kinases in the dataset = {} \n shape of a kinase input matrix [max length, alphabet size] = {}".format(N_p,shape_p))
        print(" Number of kinase families present in the data = {}".format(N_families))

    def get_dimensions(self):
        sample = self.keys_train.iloc[0]
        return list( self.drug_matrix_shape )+list( self.target_matrix_shape )+[len(self.family_token_keys)]

    def get_all_affinities(self):
        return self.keys_train['standard_value']
    
    def epoch_status(self):
        return self.BT.epoch_status()
    
    def get_progress(self):
        return self.BT.get_progress()

    def get_n_batches(self):
        return self.BT.get_n_batches()

if __name__=="__main__":
    print("This is the python file for Data tool class")

