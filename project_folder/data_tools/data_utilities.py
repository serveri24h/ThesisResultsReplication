### BIOACTIVITY SPECIFIC LIBRARIES IF AVAILABLE
PROPER_ENV = True
try:
    from chembl_webresource_client.new_client import new_client
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from Bio import SeqIO
except ImportError:
    PROPER_ENV = False

### COMMON LIBRARIES
import torch
import numpy as np
import pandas as pd
import threading
import queue
from math import ceil

### LOCAL MODULES
import helpers as h
import constants as const

#####################################
### DATA TRANSFORMATION FUNCTIONS ###
#####################################

def get_affinity(y):
    y = torch.Tensor(y)
    return torch.unsqueeze( 9-torch.log10(y), dim=1 )

def tokenize_kinase_families( kinase_family, dti):
    X = list( map(kinase_family.get, dti['target_id']) )
    family_token_keys = {}
    n = 0
    for x in X:
        if x not in family_token_keys:
            family_token_keys[x] = n
            n+=1
    return family_token_keys

def create_token_keys(data_dict):
    M = 0 # M will be the largest sample length
    token_keys = {"PAD":0} #Infere the aplhabet size of data
    idx = 1
    for molecule in list(data_dict.values()):
        l_m = len(molecule)
        if l_m > M:
            M = l_m
        for x in molecule:
            if x not in token_keys:
                token_keys[x]=idx
                idx+=1
    return token_keys, [M,idx] 

def one_hot_code_family(X,family_token_keys,warnings=True):
    n = len(family_token_keys)
    N = len(X)
    X_Family = torch.zeros(N,n)
    for i in range(N):
        try:
            X_Family[i][family_token_keys[X[i]]] = 1
        except:
            if warnings: print( "WARNING! Family \"{}\" not seen by the algorithm".format( X[i] ) )
    return X_Family

def one_hot_code_matrix(s,token_keys,m_shape,prints=True):
    X = [token_keys[ s[j] ] for j in range(len(s))]
    M = torch.zeros(m_shape)
    for i,x in enumerate(X):
            try:
                M[i,x]=1
            except Exception as e:
                print(e)
                if prints: 
                    print( "{} Token {} was not found from the keys".format(i, x) )
                    #print( "\n\n Tää1 {} \n tää2 {} \n tää3 {} \n\n".format(m_shape, token_keys, X) )
    return M

def create_integer_vector(s,token_keys):
    return torch.tensor([token_keys[c] for c in s])

def create_matrix_database(data_dict,token_keys,M_shape,prints=True):
    Matrix_DATA = {}
    for k in list(data_dict.keys()):
        Matrix_DATA[k] = one_hot_code_matrix(data_dict[k], token_keys , M_shape, prints=prints )
    return Matrix_DATA

def create_vector_database(data_dict,token_keys,prints=True):
    Vector_DATA = {}
    for k,s in list(data_dict.items()):
        Vector_DATA[k] = create_integer_vector(s,token_keys)
    return Vector_DATA

def strings_to_matrix(data_str, token_keys, M_shape, prints=True, warnings=True):
    M = []
    succ = []
    for i,s in enumerate(data_str):
        try:
            M.append( one_hot_code_matrix(s, token_keys, M_shape, prints=prints) ) 
            succ.append(i)
        except Exception as e: 
            if warnings: 
                if s is not None: 
                    s = str(s)
                    if len(s)>50: s = s[:50]+'...'
                    print("WARNING! Failed to convert {}th sample into matrix form using following string: {} with following exception:\n{}".format(i, s,e) ) 
                else:
                    print("WARNING! Failed to convert {}th sample {} into matrix form.".format(i,s))
            M.append(torch.zeros(M_shape[0],M_shape[1]))
    return torch.stack( M ), succ

def strings_to_vectors(data_str, token_keys, M_shape, prints=True, warnings=True):
    M = []
    succ = []
    for i,s in enumerate(data_str):
        try:
            M.append(create_integer_vector(s,token_keys)) 
            succ.append(i)
        except Exception as e: 
            if warnings: 
                if s is not None: 
                    s = str(s)
                    if len(s)>50: s = s[:50]+'...'
                    print("WARNING! Failed to convert {}th sample into matrix form using following string: {} with following exception:\n{}".format(i, s,e) ) 
                else:
                    print("WARNING! Failed to convert {}th sample {} into matrix form.".format(i,s))
            M.append(torch.zeros(M_shape[0],M_shape[1]))
    return M, succ

def create_finger_prints(SMILES):
    # read molecule from smiles
    fps = []
    for S in SMILES:
        # read molecule from smiles 
        mol = Chem.MolFromSmiles(S)
        # calculate rdk fingerprint
        rdk_fp = Chem.RDKFingerprint(mol, maxPath=5, fpSize=1024)
        # turn fingerprint into np array and return
        fps.append( np.array(rdk_fp) )
    return fps

def create_finger_prints_from_dict(DATA_drug):
    # read molecule from smiles
    fps = {}
    for i,S in DATA_drug.items():
        # read molecule from smiles
        try:
            mol = Chem.MolFromSmiles(S)
            # calculate rdk fingerprint
            rdk_fp = Chem.RDKFingerprint(mol, maxPath=5, fpSize=1024)
            # turn fingerprint into np array and return
        except Exception as e:
            rdk_fp = None
        fps[i] = torch.tensor(rdk_fp)
    return fps

def prepare_finger_prints(DATA=None):
    return create_finger_prints_from_dict(DATA)

################################
#######   ACCESS TARGET   ######
################################

def get_amino_acids_from_fasta(ids_target,fasta_file,prints):
    # GET AMINO ACID SEQ FOR THE PROTEIN
    data = {}
    # match protein sequence
    for seq in SeqIO.parse(fasta_file, 'fasta'): # self.src_data+
        seqId = seq.id.split('|')[1]
        fasta_seq = seq.seq._data
        if seqId in ids_target:
            if len(fasta_seq) < const.MAX_SIZE_P: 
                data[seqId] = fasta_seq
            else: 
                if prints: print('Unappropriate size {} for: '.format(len(fasta_seq)), seqId)
    return data

###################################
#######  ACCESS CHEMbl DATA  ######
###################################

def get_molecule_chunk(chunks,chunk_size,drug_dict,ids_drug,prints=True):
    while not chunks.empty():
        k = chunks.get()
        try:
            molecules = new_client.molecule.get(molecule_chembl_id=ids_drug[k*chunk_size:(k+1)*chunk_size]).only(['molecule_chembl_id','molecule_structures'])
            if prints: print("Starting chunk",k)
            for i,m in enumerate(molecules):
                try:
                    smiles = m['molecule_structures']['canonical_smiles']
                    if len(smiles) <= const.MAX_SIZE_D or len(smiles)==0:
                        drug_dict[m['molecule_chembl_id']] = smiles
                    else:
                        if prints: print('Unappropriate size for: ', m['molecule_chembl_id'])
                except:
                    if prints: print("Unable to access:", m['molecule_chembl_id'])
            if prints: print("Done with chunk",k)
        except:
            if prints: print("HOX chunk {} FAILED at {:.2f}%".format(k,i*100/chunk_size))
        chunks.task_done()

def prepare_smiles(ids_drug,prints):
    chunk_size = 1000
    n_threads = 10
    chunks = queue.Queue()
    drug_dict = {}

    l = len(ids_drug)
    for k in range(ceil(l/chunk_size)):
        chunks.put(k)
    
    for _ in range(n_threads):
        t = threading.Thread(target = get_molecule_chunk, args=[chunks,chunk_size,drug_dict,ids_drug,prints])
        t.start()
    chunks.join()
    if prints: print("Done with all chunks")
    return drug_dict

def get_id_from_chembl(data,queries=['inchi','smiles']):
    n_samples = len(data)
    data_is_df = isinstance(data, pd.core.frame.DataFrame )
    chembl_ids = []

    if data_is_df: 
        x_inchi = data['standard_inchi_key']
        x_smiles = data['SMILES']
    else:
        if isinstance(data[0], list):
            if 'inchi' in queries: x_inchi = data[:,queries.index("inchi")]
            if 'smiles' in queries: x_smiles = data[:,queries.index("smiles")]
        else:
            if 'inchi' in queries: x_inchi = data
            if 'smiles' in queries: x_smiles = data

    for n in range(n_samples):
        if 'inchi' in queries:
            try:
                chem_id = new_client.molecule.filter(molecule_structures__standard_inchi_key=x_inchi[n]).only(['molecule_chembl_id'])
                chem_id = chem_id[0]['molecule_chembl_id']
                chembl_ids.append(chem_id)
                continue
            except:
                chem_id = None
        if 'smiles' in queries:
            try:
                chem_id = new_client.molecule.filter(molecule_structures__canonical_smiles__connectivity=x_smiles[n]).only(['molecule_chembl_id'])
                chem_id = chem_id[0]['molecule_chembl_id']
            except:
                chem_id = None
        chembl_ids.append(chem_id)
    if data_is_df:
        data['compound_id'] = chembl_ids
        return data
    return chembl_ids

###################################
####### READING SPECIFIC FILES ####
###################################

def read_kinase_information_from_file( src_kinase):
    # GET KINASE INFORMATION
    kinase_info = pd.read_excel(src_kinase)
    kinase_family = kinase_info.iloc[:,[5, 7]].set_index('UniprotID').to_dict()['Family']
    return kinase_info,kinase_family

def read_training_data_from_file(src_dti,index_range=None):
    df_dti = h.load_from_csv(file_name=src_dti, index_range=index_range)#[['compound_id','target_id','standard_value','standard_type']]
    return df_dti

def read_test_data_from_file(file_name_data,sheet=None):
    read_cols = ['Compound SMILES','UniProt ID','pKd (-logM)','Compound InchiKeys']
    new_col_names = ['SMILES','target_id','standard_value','standard_inchi_key']
    dti = h.read_df_from_xlsx(file_name_data,cols=read_cols,col_names=new_col_names,sheet=sheet)
    return get_id_from_chembl(dti,queries=['inchi','smiles'])


#############################
####### FILTERING DATA ######
#############################

def filter_training_data(df_dti,kinase_info,standard_types):
    # SELECT DTIs COVERED IN KINASE-INFO
    df_dti = df_dti[df_dti.target_id.isin(set(kinase_info.UniprotID))]

    # SELECT DTIs THAT MEET THE CRITERIONS
    df_dti = df_dti.loc[(df_dti['target_id'] != None )&(df_dti['compound_id'] != None )
                &(df_dti['standard_type'].str.upper().isin(standard_types))
                &(df_dti['standard_relation'] == '=')&(df_dti['standard_value']>0)
                &(df_dti['standard_units']=='NM'),['standard_value','target_id','compound_id']]
    return df_dti.groupby(['target_id','compound_id'], as_index=False ).mean()

def filter_training_data_with_test_data(dti_train,dti_test):
    df_merge = dti_train.merge(dti_test, on=['target_id','compound_id'], how='outer', indicator=True)
    df_merge = df_merge.loc[df_merge._merge == 'left_only']
    return df_merge.rename( columns={'standard_value_x': 'standard_value'})[['standard_value','target_id','compound_id']]#.reset_index(drop=True)

