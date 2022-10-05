import numpy as np
import pandas as pd
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import constants as const
from data_tools import DTI_DATA_TOOL
from data_tools.data_tools import BATCH_TOOL
import data_tools.data_utilities as dutils

#############################################################
##########   TEST MODULES USED FOR DIAGNOSTICS   ############
#############################################################

def module_core_functions( index_range = None, prints=False, ):
    print("\n >> RUNNING THE TEST MODULE: DATA TOOLS CORE FUNCTIONS TESTING << \n")
    
    ### TEST 0 - INITIATE OBJECT
    try:
        DATA_OBJECT = DTI_DATA_TOOL(prints=prints)
    except Exception as e:
        print("Initialising the class failed with following error\n", e)
        return False
    print("* TEST 0 - PASS! Initialising the Data Tools Class was succesful!")
    
    ### TEST 1 - READ DATA FROM THE FILE
    try:
        DATA_OBJECT.read_training_data(index_range=index_range)
        assert DATA_OBJECT.keys_train is not None, "--> {} <-- is not proper data".format(DATA_OBJECT.keys_train)
    except Exception as e:
        print("Reading the training data from the file failed with following error: \n", e)
        return False
    print("* TEST 1 - PASS! Reading the training data from the file was succesful!")

    ### TEST 2 - READ KINASE INFORMATION FROM THE FILE
    try:
        DATA_OBJECT.kinase_info, DATA_OBJECT.kinase_family = dutils.read_kinase_information_from_file(src_kinase=DATA_OBJECT.src_data+DATA_OBJECT.src_kinase)
    except Exception as e:
        print("Reading kinase data from the file failed with following error: \n", e)
        return False
    print("* TEST 2 - PASS! Reading Kinase information from file was succesful")

    ### TEST 3 - FILTER THE DATA
    try:
        DATA_OBJECT.keys_train = dutils.filter_training_data(DATA_OBJECT.keys_train,DATA_OBJECT.kinase_info,DATA_OBJECT.standard_types)
        DATA_OBJECT.ids_drug = list(DATA_OBJECT.keys_train.compound_id.unique())
        DATA_OBJECT.ids_target = list(DATA_OBJECT.keys_train.target_id.unique())
    except Exception as e:
        print("Filtering the data failed with following error: \n", e)
        return False  
    print("* TEST 3 - PASS! Filtering the data was succesful")
    
    ### TEST 4 - PREPARE THE SMILES STRINGS
    try:
        DATA_OBJECT.DATA_drug = dutils.prepare_smiles(DATA_OBJECT.ids_drug,prints=DATA_OBJECT.prints)
    except Exception as e:
        print("Accessing the SMILES strings failed with following error \n", e)
        return False
    print("* TEST 4 - PASS! Accessing chemical data was succesful")

    ### TEST 5 - PREPARE THE AMINO ACID STRINGS
    try:
        DATA_OBJECT.DATA_target = dutils.get_amino_acids_from_fasta(ids_target=DATA_OBJECT.ids_target, fasta_file=DATA_OBJECT.src_data+DATA_OBJECT.src_fasta, prints=DATA_OBJECT.prints)
    except Exception as e:
        print("Accessing the amino acid seqs failed with following error \n", e)
        return False
    print("* TEST 5 - PASS! Accessing protein data was succesful")
    
    ### TEST 6 - TOKENIZE THE KINASE FAMILIES
    try:
        DATA_OBJECT.family_token_keys = dutils.tokenize_kinase_families(DATA_OBJECT.kinase_family, DATA_OBJECT.keys_train)
    except Exception as e:
        print("Tokenizing kinase families failed with following error \n", e)
        return False
    print("* TEST 6 - PASS! Tokenizing kinase families was succesful")

    ### TEST 7 - CREATE FINGERPRINTS
    try:
        DATA_OBJECT.fingerprints_train = dutils.prepare_finger_prints(DATA_OBJECT.DATA_drug)
    except Exception as e:
        print("Generating fingerprints failed with following error \n", e)
    print("* TEST 7 - PASS! Generating fingerprints was succesfull")
    
    ### TEST 8 - PREPARE DATA MATRICES 
    try:
        DATA_OBJECT.keys_train = DATA_OBJECT.keys_train.loc[(DATA_OBJECT.keys_train['compound_id'].isin(DATA_OBJECT.DATA_drug))&(DATA_OBJECT.keys_train['target_id'].isin(DATA_OBJECT.DATA_target))]
        DATA_OBJECT.ids_drug = list(DATA_OBJECT.DATA_drug.keys())
        DATA_OBJECT.ids_target = list(DATA_OBJECT.DATA_target.keys())

        DATA_OBJECT.drug_token_keys, DATA_OBJECT.drug_matrix_shape = dutils.create_token_keys(DATA_OBJECT.DATA_drug)
        DATA_OBJECT.DATA_drug_matrices = dutils.create_matrix_database(DATA_OBJECT.DATA_drug,DATA_OBJECT.drug_token_keys,DATA_OBJECT.drug_matrix_shape)
        DATA_OBJECT.target_token_keys, DATA_OBJECT.target_matrix_shape = dutils.create_token_keys(DATA_OBJECT.DATA_target)
        DATA_OBJECT.DATA_target_matrices = dutils.create_matrix_database(DATA_OBJECT.DATA_target,DATA_OBJECT.target_token_keys, DATA_OBJECT.target_matrix_shape)

        if len(DATA_OBJECT.DATA_drug_matrices.keys())!=len(DATA_OBJECT.DATA_drug.keys() or DATA_OBJECT.DATA_target_matrices.keys())!=len(DATA_OBJECT.DATA_target.keys()):
            print(len(DATA_OBJECT.DATA_drug_matrices.keys()))
            print(len(DATA_OBJECT.DATA_drug.keys()))
            print("Information lost when transforming data into one-hot-coded matrixes")
            print("DRUG:   #strings={} #matrices={}".format( len(DATA_OBJECT.DATA_drug.keys()), len(DATA_OBJECT.DATA_drug_matrices.keys()) ) )
            print("TARGET: #strings={} #matrices={}".format( len(DATA_OBJECT.DATA_target.keys()), len(DATA_OBJECT.DATA_target_matrices.keys()) ) )
            return False
    except Exception as e:
        print("Transforming data into one-hot-coded matrices failed with error: \n", e)
        return False
    print("* TEST 8 - PASS! Transforming data into one-hot-coded matrices was succesful!")
    
    ### TEST 8 - PREPARE SAMPLE TRAINING BATCH
    try:
        DATA_OBJECT.devide_datasets( p=0.25 )
    except Exception as e:
        print("Failed to divide data into training and validation sets with following error \n", e)
        return False
    try:
        DATA_OBJECT.BT = BATCH_TOOL(len(DATA_OBJECT.TRAIN))
    except Exception as e:
        print("Initiating the batch tool failed with following error \n", e)
        return False
    try:
        X_test = DATA_OBJECT.training_batch()
        assert X_test[4].shape == torch.Size([DATA_OBJECT.BT.get_batch_size(), 1])
    except Exception as e:
        print("Preparing a training batch failed with following error \n", e)
        return False
    print("* TEST 9 - PASS! Preparing a sample training batch was succesful!")
    return True

# MODULE 2 - FULL DATA PREPARATION ROUTINE

def module_prepare_full(prints=False,index_range=None):
    print("\n >> RUNNING THE TEST MODULE: FULL DATA PREPARATION ROUTINE << \n")
    ### TEST 0 - INITIATE OBJECT
    try:
        DATA_OBJECT = DTI_DATA_TOOL(prints=prints)
    except Exception as e:
        print("Initialising the class failed with following error\n", e)
        return False
    print("* TEST 0 - PASS! Initialising the Data Tools Class was succesful!")

    ### TEST 1 - RUNNING THE SUBROUTINE "prepare_full()""
    DATA_OBJECT.prepare_full(index_range=index_range)
    try:
        pass
        #DATA_OBJECT.prepare_full(index_range=index_range)
    except Exception as e:
        print("Failed to prepare full dataset with error", e)
        return False
    print("* TEST 1 - PASS! running the data preparation routine was succesful")
    return True

# MODULE 3 - SAVE AND LOAD

def module_save_and_load(prints=True,index_range=None):
    print("\n >> RUNNING THE TEST MODULE: SAVE AND LOAD TESTS << \n")
    ### TEST 0 - INITIALIZE THE DATA OBJECT WITH SAVE
    try:
        DATA_OBJECT1 = DTI_DATA_TOOL(src_intermediate=const.DUMMY_RESULTS_DIR,prints=prints)
        DATA_OBJECT2 = DTI_DATA_TOOL(src_intermediate=const.DUMMY_RESULTS_DIR,prints=prints)
    except Exception as e:
        print("Initialising the class with saving ability failed with following error:\n", e)
        return False
    print("* TEST 0 - PASS! Initialising two DataTools-Classes was succesful!")

    ### TEST 1 - READING THE TRAINING DATA AND SAVING IT TO A FILE
    try:
        DATA_OBJECT1.prepare_and_save(index_range=index_range)
        
    except Exception as e:
        print("Failed to read the full training data with error: ", e)
        return False
    print("* TEST 1 - PASS! Reading the training data and saving it into a csv-file was succesful")
    
    ### TEST 2 - READING THE sdfa
    try:
        DATA_OBJECT2.load_intermediates(src_smiles=const.ID_SMILES_FILE, src_target=const.ID_AMINO_FILE, src_fps=const.ID_FPS_FILE, 
                                        src_int_train=const.INTERMEDIATE_TRAIN_FILE ,src_int_test1=const.INTERMEDIATE_TEST_FILE1, src_int_test2=const.INTERMEDIATE_TEST_FILE2)
    except Exception as e:
        print("Failed to read the previously saved data from csv-file with following error: ", e)
        return False
    print("* TEST 2 - PASS! Reading a previously saved data from a csv-file was succesful")

    ### TEST 3 - READING THE sdfa  
    try:
        DATA_OBJECT1.prepare_from_intermediates()
    except Exception as e:
        print("EEIII", e)
        return False
    try:
        DATA_OBJECT2.prepare_from_intermediates()
    except Exception as e:
        print("NPPPP", e)
        return False
    print("* TEST 3 - PASS! Preparing the data from intermediates was succesful")   


    ### TEST 4 - READING THE sdfa
    try:
        # SMILES STRINGS MATCH WITH CHEMBL ID
        for i,x in DATA_OBJECT1.DATA_drug.items():
            assert x==DATA_OBJECT2.DATA_drug[i], "Mismatch in smiles strings with index {} \nsaved string:  {} \nloaded string: {}".format(i,x,DATA_OBJECT2.DATA_drug[i])
        for i,x in DATA_OBJECT2.DATA_drug.items():
            assert x==DATA_OBJECT1.DATA_drug[i], "Mismatch in smiles strings with index {} \nsaved string:  {} \nloaded string: {}".format(i,DATA_OBJECT1.DATA_drug[i],x)
        
        # AMINO ACID SEQS MATCH WITH UNIPROT ID
        for i,x in DATA_OBJECT1.DATA_target.items():
            assert x==DATA_OBJECT2.DATA_target[i], "Mismatch in target strings with index {} \nsaved string:  {} \nloaded string: {}".format(i,x,DATA_OBJECT2.DATA_target[i])
        for i,x in DATA_OBJECT2.DATA_target.items():
            assert x==DATA_OBJECT1.DATA_target[i], "Mismatch in target strings with index {} \nsaved string:  {} \nloaded string: {}".format(i,DATA_OBJECT1.DATA_target[i],x)

        # FINGER PRINT MATCH WITH CHEMBL ID
        for i,x in DATA_OBJECT1.fingerprints_train.items():
            assert torch.sum( (x-DATA_OBJECT2.fingerprints_train[i])**2 )==0, "Mismatch in fingerprints with index {} \nsaved string:  {} \nloaded string: {}".format(i,x,DATA_OBJECT2.DATA_drug[i])
        for i,x in DATA_OBJECT2.fingerprints_train.items():
            assert torch.sum( (x-DATA_OBJECT1.fingerprints_train[i])**2 )==0, "Mismatch in fingerprints with index {} \nsaved string:  {} \nloaded string: {}".format(i,DATA_OBJECT1.DATA_drug[i],x)
    except Exception as e:
        print("Comparison failed with error:", e)
        return False
    print("* TEST 4 - PASS! No missmatches found in comparison between loaded and saved data!")   
    return True

# MODULE 5
def module_testset_diagnostics( prints = False, index_range = None):
    print("\n >> RUNNING THE TEST MODULE: TEST DATASET TESTS << \n")
    ### TEST 0 - INITIATE OBJECT
    try:
        DATA_OBJECT = DTI_DATA_TOOL(prints=prints)
    except Exception as e:
        print("Initialising the class failed with following error\n", e)
        return False
    print("* TEST 0 - PASS! Initialising the Data Tools Class was succesful!")
    
    ### TEST 1 - RUNNING THE SUBROUTINE "prepare_full()""
    DATA_OBJECT.prepare_full(index_range=index_range)
    try:
        pass
        #DATA_OBJECT.prepare_full(index_range=index_range)
    except Exception as e:
        print("Failed to prepare full dataset with error", e)
        return False
    print("* TEST 1 - PASS! running the data preparation routine was succesful")
    ### TEST 2 - FILTERING TRAINING DATA WITH TEST DATA
    try:
        cols = ['standard_value','target_id','compound_id']
        dummy_train_dti = pd.DataFrame([[1.0,"P1","D1"], [2.0,"P1","D2"],[3.0,"P2","D2"], [2.0,"P2","D3"], [2.0,"P2","D4"]], columns=cols)
        dummy_test_dti = pd.DataFrame([[1.0,"P1","D1"],[5.0,"P2","D2"],[5.0,"P3","D3"]], columns=cols)
        true_filtered_dti = dummy_train_dti.drop([0,2])
        assert true_filtered_dti.equals(dutils.filter_training_data_with_test_data(dummy_train_dti,dummy_test_dti)), "Unexpected filtered dataframe."
    except Exception as e:
        print("Filtering the instances in test set from training set failed with error:\n", e)
        return False
    print("* TEST 2 - PASS!! Filtering was succesful!")


    ### TEST 2 - PREPARE TEST DATA 1 AND 2
    try:
        TEST1 = DATA_OBJECT.prepare_test_batch(0)
        assert TEST1 is not None, "Preparing the test set 1 returned None"
    except Exception as e:
        print("Preparing the test set 1 failed with following error:\n",e)
        return False
    try:
        TEST2 = DATA_OBJECT.prepare_test_batch(1)
        assert TEST2 is not None, "Preparing the test set 2 returned None"
    except Exception as e:
        print("Preparing the test set 2 failed with following error:\n",e)
        return False
    
    print("* TEST 3 - PASS!! Preparing Test Data was succesful!")
    return True

# MODULE 6
def module_6():
    print("\n >> RUNNING THE TEST MODULE: CUDA TESTS << \n")
    ### TEST 0 - INITIATE OBJECT
    try:
        DATA_OBJECT = DTI_DATA_TOOL()
    except Exception as e:
        print("Initialising the class failed with following error\n", e)
        return False

    ## TESTING CUDA
    if torch.cuda.is_available():
        print("Testing that CUDA works properly")
        try:
            X = DATA_OBJECT.random_batch()
            for x in X:
                assert x.de
        except:
            print("Something went wrong with cuda...")
            return False
    else:
        print("CUDA not available") 
    return True

    
def run_tests():
    # RUN MODULES
    run_core_functions = True
    run_prepare_subset = False
    run_save_and_load = False
    run_prepare_full = False
    run_testset_diagnostics = False
    run_test_cuda = False

    # DATA FILES
    
    if run_core_functions:
        if not module_core_functions(index_range=[0,40000],prints=False):
            return False
    
    if run_prepare_subset:
        if not module_prepare_full(index_range=[0,1000],prints=True):
            return False
    
    if run_save_and_load:
        if not module_save_and_load(index_range=[0,40000],prints=False):
            return False

    
    if run_prepare_full:
        if not module_prepare_full():
            return False
    
    if run_testset_diagnostics:
        if not module_testset_diagnostics(index_range=[0,40000],prints=True):
            return False
    
    if run_test_cuda:
        if not module_6():
            return False
    
    return True

if __name__=="__main__":
    if run_tests():
        print("\nALL THE TESTS PASSED!! - THIS IS A WONDERFUL DAY\n")
    else:
        print("\nSOMETHING FAILED - THERE IS A BUG IN THE CODE\n")
