import pandas as pd #########################################################
import torch ################################################################
import os                                                           #########
#######  XX   XX   XXXXXX  XX      XXXXXX   XXXXXX  XXXXXX    XXXXX   #######         
######   XX   XX   XX      XX      XX   XX  XX      XX   XX  XX   XX   ######
#####    XX   XX   XX      XX      XX   XX  XX      XX   XX  XX         #####
#####    XXXXXXX   XXXXX   XX      XXXXXX   XXXXX   XXXXXX    XXXXX     #####
#####    XX   XX   XX      XX      XX       XX      XX XX         XX    #####
######   XX   XX   XX      XX      XX       XX      XX  XX   XX   XX   ######
#######  XX   XX   XXXXXX  XXXXXX  XX       XXXXXX  XX   XX   XXXXX   #######
########                                                             ########
#############################################################################
#############################################################################
#####                                                                 #######
#####           THIS FILE CONTAINS SEVERAL HELPERS FUNCTIONS            #####
######                                                                 ######
#############################################################################
#############################################################################

def folder_name(src_folder):
    if src_folder[-1]!='/':
        src_folder+='/'
    return src_folder

def csv_name(src_csv):
    if src_csv[-1]!='/':
        src_csv+='/'
    return src_csv

def xlsx_name(src_xlsx):
    if src_xlsx[-5]!='.xlsx':
        src_xlsx+='.xlsx'
    return src_xlsx

def save_dict_to_csv(save_dict, file_name):
    df = pd.DataFrame.from_dict(save_dict, orient="index")
    if file_name[-4:] != ".csv": file_name+=".csv"
    try:
        df.to_csv(file_name)
    except Exception as e:
        print("Failed to save the data with the following exception: ", e)
    
def save_df_to_csv(df, file_name):
    if file_name[-4:] != ".csv": file_name+=".csv"
    try:
        df.to_csv(file_name)
    except Exception as e:
        print("Failed to save the data with the following exception: ", e)


def load_from_csv(file_name, data_type='dataframe', index_range=None):
    if file_name[-4:] != ".csv": file_name+=".csv"
    try:
        if index_range:
            df = pd.read_csv(file_name, skiprows=[i for i in range(1,index_range[0])],nrows=index_range[1],low_memory=False)
        else:
            df = pd.read_csv(file_name, index_col=False,low_memory=False)
        col_name = df.columns[0]
        if data_type=='dataframe': 
            return df
        elif data_type=='dict': 
            lollis_jee = pd.read_csv(file_name, header=None, index_col=0, squeeze=True)
            lollis_jee=lollis_jee.iloc[1:]
            lollis_jee=lollis_jee.to_dict()
            return lollis_jee
        else: 
            return df[col_name].tolist()
    except Exception as e:
        print("Failed to load the data from csv-file {} with the following exception: ".format(file_name), e)
        return None

def read_df_from_xlsx(file_name,sheet=None,cols=None,col_names=None):
    if file_name[-5:] != ".xlsx": file_name+=".xlsx"
    try:
        df = pd.read_excel(file_name,sheet)##+self.src_data
    except Exception as e:
        print(file_name)
        print("Failed to load the data from xlx-file inf folder {} with the following exception: \n".format(os.getcwd()), e)
        return None
    if cols is None: 
        return(df)
    else:
        df = df[cols]
        if col_names is None: 
            return df[cols]
        df = df.rename(columns={cols[i]:col_names[i] for i in range(len(col_names)) })
        return df

def combine_dicts(dict1,dict2):
    for i,x in dict2.items():
        if dict1[i] is None:
            dict1[i] = x
        elif dict1[i] != x:
            print("HOX! MISMATCH ENCOUTERED BETWEEN DICTS")


def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dev_count = 1
        #dev_count = torch.device_count()
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        dev_count = 1
        print("Running on the CPU")
    return device, dev_count