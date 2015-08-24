import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
#from sklearn.metrics import confusion_matrix

from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM
from sklearn import svm, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import gzip
from random import randint

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, AutoEncoder
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import containers, normalization
#from sknn.mlp import Classifier, Layer
import pdb
import os
import sys
import random

from theano import tensor as T
from keras import regularizers
from keras.optimizers import kl_divergence
'''
class SparseActivityRegularizer(regularizers):
    def __init__(self, l1=0., l2=0., p=0.05):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        p_hat = T.sum(T.mean(self.layer.get_output(True) ** 2, axis=0))
        loss += kl_divergence(self.p, p_hat)
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "p": self.l1}
'''
        
def run_struct_classifier():
    abc = "abcdefghijklmnopqrstuvwxyz"
    
    letters = load_letters()
    X, y, folds = letters['data'], letters['labels'], letters['folds']
    # we convert the lists to object arrays, as that makes slicing much more
    # convenient
    X, y = np.array(X), np.array(y)
    X_train, X_test = X[folds == 1], X[folds != 1]
    y_train, y_test = y[folds == 1], y[folds != 1]
    
    # Train linear SVM
    svm = LinearSVC(dual=False, C=.1)
    # flatten input
    svm.fit(np.vstack(X_train), np.hstack(y_train))
    
    # Train linear chain CRF
    model = ChainCRF()
    ssvm = OneSlackSSVM(model=model, C=.1, inference_cache=50, tol=0.1)
    ssvm.fit(X_train, y_train)
    
    print("Test score with chain CRF: %f" % ssvm.score(X_test, y_test))
    
    print("Test score with linear SVM: %f" % svm.score(np.vstack(X_test),
                                                       np.hstack(y_test)))
    
    # plot some word sequenced
    n_words = 4
    rnd = np.random.RandomState(1)
    selected = rnd.randint(len(y_test), size=n_words)
    max_word_len = max([len(y_) for y_ in y_test[selected]])
    fig, axes = plt.subplots(n_words, max_word_len, figsize=(10, 10))
    fig.subplots_adjust(wspace=0)
    for ind, axes_row in zip(selected, axes):
        y_pred_svm = svm.predict(X_test[ind])
        y_pred_chain = ssvm.predict([X_test[ind]])[0]
        for i, (a, image, y_true, y_svm, y_chain) in enumerate(
                zip(axes_row, X_test[ind], y_test[ind], y_pred_svm, y_pred_chain)):
            #a.matshow(image.reshape(16, 8), cmap=plt.cm.Greys)
            a.text(0, 3, abc[y_true], color="#00AA00", size=25)
            a.text(0, 14, abc[y_svm], color="#5555FF", size=25)
            a.text(5, 14, abc[y_chain], color="#FF5555", size=25)
            a.set_xticks(())
            a.set_yticks(())
        for ii in xrange(i + 1, max_word_len):
            axes_row[ii].set_visible(False)
    
    plt.matshow(ssvm.w[26 * 8 * 16:].reshape(26, 26))
    plt.title("Transition parameters of the chain CRF.")
    plt.xticks(np.arange(25), abc)
    plt.yticks(np.arange(25), abc)
    plt.show()
      
''' 
echo -e "3I5F\n2p4k\n2p4m" | while read I; do curl -s "http://www.rcsb.org/pdb/rest/customReport?pdbids=${I}&customReportColumns=structureId,chainId,entityId,sequence,db_id,
db_name&service=wsdisplay&format=csv"; done >result.csv

$ echo -e "3I5F\n2p4k\n2p4m" | while read I; do curl -s "http://www.rcsb.org/pdb/rest/customReport?pdbids=${I}
&customReportColumns=structureId,chainId,entityId,sequence,db_id,db_name&service=wsdisplay&format=text" 
| xsltproc stylesheet.xsl - ; done | fold -w 80   
def deep_learning_classifier(X_train, y_train):
    nn = Classifier(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Linear")],
    learning_rate=0.02,
    n_iter=10)
    nn.fit(X_train, y_train)
    
    y_valid = nn.predict(X_valid)
    
    score = nn.score(X_test, y_test)
'''    
def get_uniq_pdb_protein_rna():
    protein_set = set()
    with open('ncRNA-protein/NegativePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            protein_set.add(pro1.split('-')[0])
            protein_set.add(pro2.split('-')[0])
    
    with open('ncRNA-protein/PositivePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            protein_set.add(pro1.split('-')[0])
            protein_set.add(pro2.split('-')[0])
    return protein_set  

def download_seq_from_PDB(protein_set, outfile_name):
    fw = open(outfile_name, 'w')
    for val in protein_set:
        cli_str = 'curl -s "http://www.rcsb.org/pdb/rest/customReport?pdbids='+ val +'&customReportColumns=structureId,chainId,sequence&service=wsdisplay&format=csv" >ncRNA-protein/tmpseq.csv'        
        cli_fp = os.popen(cli_str, 'r')
        cli_fp.close()
        #pdb.set_trace()
        f_in = open('ncRNA-protein/tmpseq.csv', 'r')
        for line in f_in:
            values = line.rstrip().split('<br />')
            for val in values:
                if 'structureId' in val:
                    continue
                if len(val) ==0:
                    continue
                pdbid, chainid, seq = val.split(',')
                fasa_name = pdbid[1:-1] + '-' + chainid[1:-1]
                fw.write('>' + fasa_name + '\n')
                fw.write(seq[1:-1] + '\n')
                #pdb.set_trace()
        f_in.close()
    fw.close()
    
def get_all_PDB_id():
    protein_set =  get_uniq_pdb_protein_rna()
    download_seq_from_PDB(protein_set, 'ncRNA-protein/all_seq.fa')

def get_protein_rna_id(inputfile):
    protein_set = set()
    with open(inputfile, 'r') as fp:
        for line in fp:
            if line[0] == '#':
                continue
            else:
                protein, rna = line.rstrip('\r\n').split()
                protein_set.add(protein.split('_')[0])
                protein_set.add(rna.split('_')[0])
    return protein_set

def judge_RNA_protein(seq):
    if 'U' in seq:
        return 'RNA'
    if all([c in 'AUGCTIN' for c in seq]):
        return 'RNA'
    else:
        return 'PROTEIN'

def generate_negative_samples_RPI2241_RPI369(seq_fasta, interaction_file, whole_file):
    seq_dict = read_fasta_file(seq_fasta)
    name_strand = {}
    type_dict = {}
    for key, tmpseq in seq_dict.iteritems():
        val, strand = key.split('-')
        seqtype = judge_RNA_protein(tmpseq)
        name_strand.setdefault(val, []).append(key)
        type_dict[key] = seqtype
        
    fw = open(whole_file, 'w')    
    existing_postive = set()
    with open(interaction_file, 'r') as fp:
        for line in fp:
            if line[0] == '#':
                continue
            pro1, pro2 = line.rstrip().split('\t')
            pro1 = pro1.replace('_', '-').upper()
            pro2 = pro2.replace('_', '-').upper()
            if type_dict[pro1] == 'RNA' and type_dict[pro2] == 'PROTEIN':
                existing_postive.add((pro2, pro1))
                fw.write(pro2 + '\t' + pro1 + '\t' + '1' + '\n')
            else:
                existing_postive.add((pro1, pro2))
                fw.write(pro1 + '\t' + pro2 + '\t' + '1' + '\n')
    
    #generate negative samples
    all_pairs = list(existing_postive)
    num_posi = len(all_pairs)
    nega_list  = []
    for val in all_pairs:
        pro1, pro2 = val
        for i in range(50):
            pro2_pare = pro2.split('-')[0]
            if name_strand.has_key(pro2_pare):
                for val in name_strand[pro2_pare]:
                    if type_dict[val] == 'PROTEIN' and val != pro1:
                        new_sele = (val, pro2)
                        if new_sele not in existing_postive:
                            #fw.write(val + '\t' + pro2 + '\t' + '0' + '\n')
                            nega_list.append(new_sele)
                            existing_postive.add(new_sele)
    
    random.shuffle(nega_list)
    for val in nega_list[:num_posi]:
        fw.write(val[0] + '\t' + val[1] + '\t' + '0' + '\n')
    fw.close()
        
    #protein_set.add(pro1.replace('_', '-').upper())
            #RNA_set.add(pro2.replace('_', '-').upper())


def get_RNA_protein_RPI2241_RPI369(seq_fasta, interaction_file, protein_file, rna_file):
    seq_dict = read_fasta_file(seq_fasta)
    RNA_set = set()
    protein_set = set()
    with open(interaction_file, 'r') as fp:
        for line in fp:
            if line[0] == '#':
                continue
            pro1, pro2, label = line.rstrip().split('\t')
            #protein_set.add(pro1.replace('_', '-').upper())
            #RNA_set.add(pro2.replace('_', '-').upper())
            protein_set.add(pro1)
            RNA_set.add(pro2)
            
    fw_pro = open(protein_file, 'w')
    for val in protein_set:
        if seq_dict.has_key(val):
            fw_pro.write('>' + val + '\n')
            fw_pro.write(seq_dict[val] + '\n')
        else:
            print val
    fw_pro.close()
    
    fw_pro = open(rna_file, 'w')
    for val in RNA_set:
        if seq_dict.has_key(val):
            fw_pro.write('>' + val + '\n')
            fw_pro.write(seq_dict[val].replace('N', '') + '\n')
        else:
            print val
    fw_pro.close()                 

def get_RPI2241_RPI369_seq():
    print 'downloading seqs'
    protein_set =  get_protein_rna_id('ncRNA-protein/RPI2241.txt')
    download_seq_from_PDB(protein_set, 'ncRNA-protein/RPI2241.fa')
    protein_set =  get_protein_rna_id('ncRNA-protein/RPI369.txt')
    download_seq_from_PDB(protein_set, 'ncRNA-protein/RPI369.fa')

def get_RPI2241_RPI369_ind_file():
    get_RNA_protein_RPI2241_RPI369('ncRNA-protein/RPI2241.fa', 'ncRNA-protein/RPI2241_all.txt', 'ncRNA-protein/RPI2241_protein.fa', 'ncRNA-protein/RPI2241_rna.fa')
    get_RNA_protein_RPI2241_RPI369('ncRNA-protein/RPI369.fa', 'ncRNA-protein/RPI369_all.txt', 'ncRNA-protein/RPI369_protein.fa', 'ncRNA-protein/RPI369_rna.fa')
    
def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = open(fasta_file, 'r')
    name = ''
    #pdb.set_trace()
    for line in fp:
        #let's discard the newline at the end (if any)
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[1:].upper() #discarding the initial >
            seq_dict[name] = ''
        else:
            #it is sequence
            seq_dict[name] = seq_dict[name] + line
    fp.close()
    
    return seq_dict
    
def get_RNA_protein():
    seq_dict = read_fasta_file('ncRNA-protein/all_seq.fa')
    RNA_set = set()
    protein_set = set()
    with open('ncRNA-protein/NegativePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            protein_set.add(pro1)
            RNA_set.add(pro2)
    
    with open('ncRNA-protein/PositivePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            protein_set.add(pro1)
            RNA_set.add(pro2) 
    
    fw_pro = open('ncRNA-protein/protein_seq.fa', 'w')
    for val in protein_set:
        if seq_dict.has_key(val):
            fw_pro.write('>' + val + '\n')
            fw_pro.write(seq_dict[val] + '\n')
        else:
            print val
    fw_pro.close()
    
    fw_pro = open('ncRNA-protein/RNA_seq.fa', 'w')
    for val in RNA_set:
        if seq_dict.has_key(val):
            fw_pro.write('>' + val + '\n')
            fw_pro.write(seq_dict[val] + '\n')
        else:
            print val
    fw_pro.close() 

def read_name_from_fasta(fasta_file):
    name_list = []
    fp = open(fasta_file, 'r')
    for line in fp:
        if line[0] == '>':
            name = line.rstrip('\r\n')[1:]
            name_list.append(name.upper())
    fp.close()
    return name_list

def get_noncode_seq():
    ncRNA_seq_dict = {}
    head  = True
    name = ''
    #pdb.set_trace()
    with open('ncRNA-protein/ncrna_NONCODE[v3.0].fasta', 'r') as fp:
        for line in fp:
            if head:
                head =False
                continue
            line = line.rstrip()
            if line == 'sequence':
                continue
            if line[0] == '>':
                name1 = line.split('|')
                name = name1[0][1:].strip()
                ncRNA_seq_dict[name] = ''
            else:
                #it is sequence
                ncRNA_seq_dict[name] = ncRNA_seq_dict[name] + line
            
    return ncRNA_seq_dict

def get_npinter_protein_seq():
    pro_dict = {}
    target_dir = 'ncRNA-protein/uniprot_seq/'
    files = os.listdir(target_dir)
    for file_name in files:
        protein_name = file_name.split('.')[0]
        with open(target_dir + file_name, 'r') as fp:
            for line in fp:
                line = line.rstrip()
                if line[0] == '>':
                    pro_dict[protein_name] = ''
                else:
                    pro_dict[protein_name] = pro_dict[protein_name] + line
                    
    return pro_dict   

def read_RNA_pseaac_fea(name_list, pseaac_file='ncRNA-protein/RNA_pse.csv'):
    print pseaac_file
    data = {}
    fp = open(pseaac_file, 'r')
    index = 0
    for line in fp:
        values = line.rstrip('\r\n').split(',')
        data[name_list[index]] = [float(val) for val in values]
        index = index + 1
    fp.close()
    return data      
    
def read_RNA_graph_feature(name_list, graph_file='ncRNA-protein/RNA_seq.gz.feature', fea_imp = None):
    print graph_file
    data = {}
    fea_len = 32768
    fp = open(graph_file, 'r')
    index = 0
    for line in fp:
        tmp_data = [0] * fea_len
        values = line.split()
        for value in values:
            val = value.split(':')
            tmp_data[int(val[0])] = float(val[1])
        if fea_imp is None:
            data[name_list[index]] = tmp_data
        else:
            data[name_list[index]] = [tmp_data[val] for val in fea_imp]
        index = index + 1
    fp.close()
    return data   

def read_protein_feature(protein_fea_file = 'ncRNA-protein/trainingSetFeatures.csv'):
    print protein_fea_file
    feature_dict = {}
    df = pd.read_csv(protein_fea_file)
    X = df.values.copy()
    for val in X:
        feature_dict[val[0].upper()] = val[2:].tolist()
    
    #pdb.set_trace()
    return feature_dict

def read_lncRNA_protein_feature(protein_fea_file = 'ncRNA-protein/trainingSetFeatures.csv'):
    print protein_fea_file
    feature_dict = {}
    df = pd.read_csv(protein_fea_file)
    X = df.values.copy()
    for val in X:
        feature_dict[val[0]] = val[2:].tolist()
    
    #pdb.set_trace()
    return feature_dict

def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**4
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com   
           
def get_4_nucleotide_composition(tris, seq):
    seq_len = len(seq)
    tri_feature = []
    for val in tris:
        num = seq.count(val)
        tri_feature.append(float(num)/seq_len)
    return tri_feature

def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group) #Alphabetically sorted list
        for c in g_members:
            # print('c' + str(c))
            # print('g_members[0]' + str(g_members[0]))
            result[c] = str(tar_list[index]) #K:V map, use group's first letter as represent.
        index = index + 1
    return result

def translate_sequence (seq, TranslationDict):
    '''
    Given (seq) - a string/sequence to translate,
    Translates into a reduced alphabet, using a translation dict provided
    by the TransDict_from_list() method.
    Returns the string/sequence in the new, reduced alphabet.
    Remember - in Python string are immutable..

    '''
    import string
    from_list = []
    to_list = []
    for k,v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    # TRANS_seq = seq.translate(str.maketrans(zip(from_list,to_list)))
    TRANS_seq = seq.translate(string.maketrans(str(from_list), str(to_list)))
    #TRANS_seq = maketrans( TranslationDict, seq)
    return TRANS_seq

def get_protein_trids(seq, group_dict):

    #protein='MQNEEDACLEAGYCLGTTLSSWRLHFMEEQSQSTMLMGIGIGALLTLAFVGIFFFVYRR'
    tran_seq = translate_sequence (seq, group_dict)
    #pdb.set_trace()
    return tran_seq

def get_3_protein_trids():
    nucle_com = []
    chars = ['0', '1', '2', '3', '4', '5', '6']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    return  nucle_com


def get_NPinter_interaction():
    RNA_set = set()
    protein_set = set()
    with open('ncRNA-protein/NPInter10412_dataset.txt', 'r') as fp:
        head  = True
        for line in fp:
            if head:
                head = False
                continue
            pro1, pro1_len, pro2, pro2_len, org = line.rstrip().split('\t')
            protein_set.add(pro2)
            RNA_set.add(pro1)
    pro_dict = get_npinter_protein_seq()        
    fw_pro = open('ncRNA-protein/NPinter_protein_seq.fa', 'w')
    for val in protein_set:
        if pro_dict.has_key(val):
            fw_pro.write('>' + val + '\n')
            fw_pro.write(pro_dict[val] + '\n')
        else:
            print val
    fw_pro.close()
    ncRNA_dict = get_noncode_seq()
    fw_pro = open('ncRNA-protein/NPinter_RNA_seq.fa', 'w')
    for val in RNA_set:
        if ncRNA_dict.has_key(val):
            fw_pro.write('>' + val + '\n')
            seq = ncRNA_dict[val].replace('T', 'U')
            #seq = seq.replace('N', '')get_RPI2241_RPI369_seq()
            fw_pro.write( seq + '\n')
        else:
            print val
    fw_pro.close()     

def get_own_lncRNA_protein(datafile = 'ncRNA-protein/lncRNA-protein-694.txt'):
    protein_seq = {}
    RNA_seq = {}
    interaction_pair = {}
    with open(datafile, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label = values[1]
                name = values[0].split('_')
                protein = name[0] + '-' + name[1]
                RNA = name[0] + '-' + name[2]
                if label == 'interactive':
                    interaction_pair[(protein, RNA)] = 1
                else:
                    interaction_pair[(protein, RNA)] = 0
                index  = 0
            else:
                seq = line[:-1]
                if index == 0:
                    protein_seq[protein] = seq
                else:
                    RNA_seq[RNA] = seq
                index = index + 1
    pdb.set_trace()
    fw = open('ncRNA-protein/lncRNA_protein.fa', 'w')
    for key, val in protein_seq.iteritems():
        fw.write('>' + key + '\n')
        fw.write(val + '\n')
        
    fw.close()
    
    fw = open('ncRNA-protein/lncRNA_RNA.fa', 'w')
    for key, val in RNA_seq.iteritems():
        fw.write('>' + key + '\n')
        fw.write(val.replace('N', '') + '\n')
        
    fw.close()
    '''
    cli_str = "python ProFET/ProFET/feat_extract/pipeline.py --trainingSetDir 'ncRNA-protein/lncRNA-protein/' \
    --trainFeatures True --resultsDir 'ncRNA-protein/lncRNA-protein/' --classType file"
    fcli = os.popen(cli_str, 'r')
    fcli.close()
    '''
def read_name_from_lncRNA_fasta(fasta_file):
    name_list = []
    fp = open(fasta_file, 'r')
    for line in fp:
        if line[0] == '>':
            name = line.rstrip('\r\n')[1:]
            name_list.append(name)
    fp.close()
    return name_list

def prepare_lncRNA_protein_feature(protein_fea_file = 'ncRNA-protein/lncRNA_trainingSetFeatures.csv', extract_only_posi = False, pseaac_file = None):
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    with open('ncRNA-protein/lncRNA-protein-694.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label = values[1]
                name = values[0].split('_')
                protein = name[0] + '-' + name[1]
                RNA = name[0] + '-' + name[2]
                if label == 'interactive':
                    interaction_pair[(protein, RNA)] = 1
                else:
                    interaction_pair[(protein, RNA)] = 0
                index  = 0
            else:
                seq = line[:-1]
                if index == 0:
                    protein_seq_dict[protein] = seq
                else:
                    RNA_seq_dict[RNA] = seq
                index = index + 1
    name_list = read_name_from_lncRNA_fasta('ncRNA-protein/lncRNA_RNA.fa')
    RNA_fea_dict = read_RNA_pseaac_fea(name_list, pseaac_file= 'ncRNA-protein/lncRNA_own.csv')            
    protein_fea_dict = read_lncRNA_protein_feature(protein_fea_file =protein_fea_file)             
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    train = []
    label = []
    chem_fea = []
    for key, val in interaction_pair.iteritems():
        protein, RNA = key[0], key[1]
        #pdb.set_trace()
        if RNA_seq_dict.has_key(RNA) and protein_seq_dict.has_key(protein) and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
            label.append(val)
            seqs = RNA_seq_dict[RNA]
            RNA_tri_fea = get_4_nucleotide_composition(tris, seqs)
            protein_seq = translate_sequence (protein_seq_dict[protein], group_dict)
            protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq)
            #RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
            #tmp_fea = protein_fea_dict[protein] + tri_fea #+ RNA_fea_dict[RNA]
            tmp_fea = protein_tri_fea + RNA_tri_fea
            train.append(tmp_fea)
            chem_fea.append(protein_fea_dict[protein] + RNA_fea_dict[RNA])
        else:
            print RNA, protein   
    
    return np.array(train), label, np.array(chem_fea)              
        
def prepare_RPI2241_369_feature(protein_fea_file, pseaac_file, rna_fasta_file, data_file, protein_fasta_file, extract_only_posi = False, graph = False):
    protein_fea_dict = read_protein_feature(protein_fea_file =protein_fea_file) 
    name_list = read_name_from_fasta(rna_fasta_file)
    seq_dict = read_fasta_file(rna_fasta_file)
    protein_seq_dict = read_fasta_file(protein_fasta_file)
    #fea_imp = keep_important_features_for_graph(keep_num=500)
    RNA_fea_dict = read_RNA_pseaac_fea(name_list, pseaac_file= pseaac_file)
    #pdb.set_trace()
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    tris = get_4_trids()
    train = []
    label = []
    chem_fea = []
    #posi_set = set()
    #pro_set = set()
    with open(data_file, 'r') as fp:
        for line in fp:
            if line[0] == '#':
                continue
            protein, RNA, tmplabel = line.rstrip('\r\n').split('\t')
            #RNA = RNA.replace('_', '-').upper()
            #protein = protein.replace('_', '-').upper()
            #posi_set.add((RNA, protein))
            #pro_set.add(protein)
            #pdb.set_trace()
            if seq_dict.has_key(RNA) and protein_seq_dict.has_key(protein) and protein_fea_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
                label.append(int(tmplabel))
                seqs = seq_dict[RNA]
                RNA_tri_fea = get_4_nucleotide_composition(tris, seqs)
                protein_seq = translate_sequence (protein_seq_dict[protein], group_dict)
                protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq)
                #RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                #tmp_fea = protein_fea_dict[protein] + tri_fea #+ RNA_fea_dict[RNA]
                tmp_fea = protein_tri_fea + RNA_tri_fea
                train.append(tmp_fea)
                chem_fea.append(protein_fea_dict[protein] + RNA_fea_dict[RNA])
            else:
                print RNA, protein
    '''
    if not extract_only_posi:
        pro_list = list(pro_set)   
        total_pro_len = len(pro_list)       
        # get negative data
        with open(data_file, 'r') as fp:
            for line in fp:
                if line[0] == '#':
                    continue
                protein, RNA = line.rstrip('\r\n').split()
                RNA = RNA.replace('_', '-').upper()
                protein = protein.replace('_', '-').upper()
                for val in range(50):
                    random_choice = randint(0,total_pro_len-1)
                    select_pro = pro_list[random_choice]
                    selec_nega= (RNA, select_pro)
                    if selec_nega not in posi_set:
                        posi_set.add(selec_nega)
                        print selec_nega
                        break
                        
                if RNA_fea_dict.has_key(RNA) and protein_fea_dict.has_key(select_pro):
                    label.append(0)
                    seqs = seq_dict[RNA]
                    tri_fea = get_4_nucleotide_composition(tris, seqs)
                    #RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                    tmp_fea =  protein_fea_dict[select_pro] + RNA_fea_dict[RNA] + tri_fea
                    train.append(tmp_fea)
                else:
                    print RNA, protein    
        #for key, val in RNA_fea_dict.iteritems():
       '''     
            
    return np.array(train), label, np.array(chem_fea)
    
    #return RNA_set, protein_set
def prepare_NPinter_feature(extract_only_posi = False, graph = False):
    print 'NPinter data'
    protein_fea_dict = read_protein_feature(protein_fea_file ='ncRNA-protein/NPinter_trainingSetFeatures.csv') 
    name_list = read_name_from_fasta('ncRNA-protein/NPinter_RNA_seq.fa')
    seq_dict = read_fasta_file('ncRNA-protein/NPinter_RNA_seq.fa')
    #fea_imp = keep_important_features_for_graph(keep_num=500)
    '''if graph == True:
        RNA_fea_dict = read_RNA_graph_feature(name_list, graph_file='ncRNA-protein/npinter_RNA_seq.gz.feature')
    else:
        
    '''
    RNA_fea_dict = read_RNA_pseaac_fea(name_list, pseaac_file='ncRNA-protein/NPInter_RNA_pse.csv')
    protein_seq_dict = read_fasta_file('ncRNA-protein/NPinter_protein_seq.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    #pdb.set_trace()
    train = []
    label = []
    chem_fea = []
    posi_set = set()
    pro_set = set()
    tris = get_4_trids()
    with open('ncRNA-protein/NPInter10412_dataset.txt', 'r') as fp:
        head  = True
        for line in fp:
            if head:
                head = False
                continue
            RNA, RNA_len, protein, protein_len, org = line.rstrip().split('\t')
            RNA = RNA.upper()
            protein = protein.upper()
            posi_set.add((RNA, protein))
            pro_set.add(protein)
            if seq_dict.has_key(RNA) and protein_fea_dict.has_key(protein) and protein_seq_dict.has_key(protein) and RNA_fea_dict.has_key(RNA):
                label.append(1)
                #RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                seqs = seq_dict[RNA]
                tri_fea = get_4_nucleotide_composition(tris, seqs)
                protein_seq = translate_sequence (protein_seq_dict[protein], group_dict)
                protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq)
                #tmp_fea = protein_fea_dict[protein] + RNA_fea_dict[RNA] + tri_fea
                #tmp_fea = protein_fea_dict[protein] + tri_fea
                tmp_fea = protein_tri_fea + tri_fea
                train.append(tmp_fea)
                chem_fea.append(protein_fea_dict[protein] + RNA_fea_dict[RNA])
            else:
                print RNA, protein
    
    if not extract_only_posi:
        pro_list = list(pro_set)   
        total_pro_len = len(pro_list)       
        # get negative data
        with open('ncRNA-protein/NPInter10412_dataset.txt', 'r') as fp:
            head  = True
            for line in fp:
                if head:
                    head = False
                    continue
                RNA, RNA_len, protein, protein_len, org = line.rstrip().split('\t')
                RNA = RNA.upper()
                protein = protein.upper()
                for val in range(50):
                    random_choice = randint(0,total_pro_len-1)
                    select_pro = pro_list[random_choice]
                    selec_nega= (RNA, select_pro)
                    if selec_nega not in posi_set:
                        posi_set.add(selec_nega)
                        #print selec_nega
                        break
                        
                if seq_dict.has_key(RNA) and protein_fea_dict.has_key(select_pro) and protein_seq_dict.has_key(select_pro) and RNA_fea_dict.has_key(RNA):
                    label.append(0)
                    #RNA_fea = [RNA_fea_dict[RNA][ind] for ind in fea_imp]
                    seqs = seq_dict[RNA]
                    tri_fea = get_4_nucleotide_composition(tris, seqs)
                    protein_seq = translate_sequence (protein_seq_dict[select_pro], group_dict)
                    protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq)
                    #tmp_fea =  protein_fea_dict[select_pro] + RNA_fea_dict[RNA] + tri_fea
                    tmp_fea = protein_tri_fea + tri_fea
                    train.append(tmp_fea)
                    chem_fea.append(protein_fea_dict[select_pro] + RNA_fea_dict[RNA])
                else:
                    print RNA, protein    
        #for key, val in RNA_fea_dict.iteritems():
            
            
    return np.array(train), label, np.array(chem_fea)

def prepare_feature(graph = False):
    print 'RPI-Pred data'
    name_list = read_name_from_fasta('ncRNA-protein/RNA_seq.fa')
    #fea_imp = keep_important_features_for_graph(keep_num=500)
    '''if graph:
        RNA_fea_dict = read_RNA_graph_feature(name_list) 
    '''
    RNA_fea_dict = read_RNA_pseaac_fea(name_list)
    
    #
    seq_dict = read_fasta_file('ncRNA-protein/RNA_seq.fa')
    protein_seq_dict = read_fasta_file('ncRNA-protein/protein_seq.fa')
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()
    
    protein_fea_dict = read_protein_feature() 
    tris = get_4_trids()
    #pdb.set_trace()
    train = []
    label = []
    chem_fea = []
    #pdb.set_trace()
    with open('ncRNA-protein/PositivePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            pro1 = pro1.upper()
            pro2 = pro2.upper()
            if protein_fea_dict.has_key(pro1) and seq_dict.has_key(pro2) and protein_seq_dict.has_key(pro1) and RNA_fea_dict.has_key(pro2):
                label.append(1)
                seqs = seq_dict[pro2]
                tri_fea = get_4_nucleotide_composition(tris, seqs)
                protein_seq = translate_sequence (protein_seq_dict[pro1], group_dict)
                protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq)
                tmp_fea = protein_tri_fea + tri_fea #+ RNA_fea_dict[pro2]
                train.append(tmp_fea)
                chem_fea.append(protein_fea_dict[pro1] + RNA_fea_dict[pro2])
            else:
                #pdb.set_trace()
                print pro1, pro2
    with open('ncRNA-protein/NegativePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            pro1 = pro1.upper()
            pro2 = pro2.upper()            
            if protein_fea_dict.has_key(pro1) and seq_dict.has_key(pro2) and protein_seq_dict.has_key(pro1) and RNA_fea_dict.has_key(pro2):
                label.append(0)
                seqs = seq_dict[pro2]
                tri_fea = get_4_nucleotide_composition(tris, seqs)
                protein_seq = translate_sequence (protein_seq_dict[pro1], group_dict)
                protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq)
                tmp_fea = protein_tri_fea + tri_fea
                #tmp_fea = protein_fea_dict[pro1] + tri_fea #RNA_fea_dict[pro2]
                train.append(tmp_fea)
                chem_fea.append(protein_fea_dict[pro1] + RNA_fea_dict[pro2])
            else:
                print pro1, pro2
    #pdb.set_trace()
    return np.array(train), label, np.array(chem_fea)



'''def get_shape_feature(shape_fea_file):
    with open(shape_fea_file, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split(',')
            
'''
def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp)
    sensitivity = float(tp)/ (tp+fn)
    specificity = float(tn)/(tn + fp)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def plot_feature_importance(importance):
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('feature_importance_xgb.png')
                

def calculate_performace_without_MCC(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    #precision = float(tp)/(tp+ fp)
    sensitivity = float(tp)/ (tp+fn)
    return acc, sensitivity

def pca_reduce_dimension(group_data, n_components = 50):
    print 'running PCA'
    pca = PCA(n_components=n_components)
    pca.fit(group_data)
    group_data = pca.transform(group_data)
    return group_data

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)  # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def get_data(datatype):
    if datatype == 'RPI-Pred':
        X, labels, chem_fea = prepare_feature(graph = False) # load_data('train.csv', train=True)
    elif datatype == 'NPInter':
        X, labels, chem_fea = prepare_NPinter_feature(graph = False)
    elif datatype == 'RPI2241':
        X, labels, chem_fea = prepare_RPI2241_369_feature('ncRNA-protein/RPI2241_trainingSetFeatures.csv', 'ncRNA-protein/RPI2241.csv', 
                                                  'ncRNA-protein/RPI2241_rna.fa', 'ncRNA-protein/RPI2241_all.txt', 'ncRNA-protein/RPI2241_protein.fa', graph = False)
    elif datatype == 'RPI369':
        X, labels, chem_fea = prepare_RPI2241_369_feature('ncRNA-protein/RPI369_trainingSetFeatures.csv', 'ncRNA-protein/RPI369.csv', 
                                                  'ncRNA-protein/RPI369_rna.fa', 'ncRNA-protein/RPI369_all.txt', 'ncRNA-protein/RPI369_protein.fa', graph = False)
    elif datatype == 'lncRNA-protein':
        X, labels, chem_fea = prepare_lncRNA_protein_feature()
        
    print X.shape
    #pdb.set_trace()
    X, scaler = preprocess_data(X)
    
    chem_fea, newscale = preprocess_data(chem_fea)
    
    dims = X.shape[1]
    print(dims, 'dims')
    
    return X, labels, chem_fea


def deep_classifier_keras(datatype = 'RPI-Pred'):
    X, labels, chem_fea = get_data(datatype)
    #y = np.array(labels)
    #X_train, X_test, b_train, b_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #X_test, ids = load_data('test.csv', train=False)
    #X_test, _ = preprocess_data(X_test, scaler)
    y, encoder = preprocess_labels(labels)
    
    num_cross_val = 5
    all_performance = []
    for fold in range(num_cross_val):
        train = []
        test = []
        train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
        test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        chem_train = np.array([x for i, x in enumerate(chem_fea) if i % num_cross_val != fold])
        chem_test = np.array([x for i, x in enumerate(chem_fea) if i % num_cross_val == fold])
        
        print("Building deep learning model...")
        
        model = Sequential()
        num_hidden = 128
        
        model.add(Dense(train.shape[1], num_hidden, init='uniform', activation='tanh'))
        #model.add(Activation('relu'))
        #model.add(PReLU((128,)))
        #model.add(BatchNormalization((128,)))
        model.add(Dropout(0.5))
        model.add(Dense(num_hidden, 128, init='uniform', activation='tanh'))
        #model.add(PReLU((128,)))
        #model.add(BatchNormalization((128,)))
        #model.add(Activation('relu'))
        model.add(Dropout(0.5))
        #model.add(Dense(128, 128, init='uniform', activation='tanh'))
        #model.add(Dropout(0.5))
        model.add(Dense(128, train_label.shape[1], init='uniform', activation='softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #sgd = SGD()
        #sgd = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=sgd) #"rmsprop")
        #model.fit(np.array(train), np.array(train_label), nb_epoch=20, batch_size=128, validation_split=0.15)
        '''
        model = Sequential()
        model.add(Dense(dims, 256, init='glorot_uniform'))
        model.add(PReLU((256,)))
        model.add(BatchNormalization((256,)))
        model.add(Dropout(0.5))
        
        model.add(Dense(256, 256, init='glorot_uniform'))
        model.add(PReLU((256,)))
        model.add(BatchNormalization((256,)))
        model.add(Dropout(0.5))
        
        model.add(Dense(256, 256, init='glorot_uniform'))
        model.add(PReLU((256,)))
        model.add(BatchNormalization((256,)))
        model.add(Dropout(0.5))
        
        model.add(Dense(256, nb_classes, init='glorot_uniform'))
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer="adam")
        '''
        print("Training model...")
        
        model.fit(train, train_label, nb_epoch=100, batch_size=100, verbose=0 )#, validation_split=0.15)
        #print model.get_weights()
        #model.set_weights(np.array([np.random.uniform(size=k.shape) for k in model.get_weights()]))
        #pdb.set_trace()
        print("Generating submission...")
        #pdb.set_trace()
        proba = model.predict_classes(test)
        
        #for pred, real in zip(proba, test_label):
        #    if real[0] == 1 and pred == 0:
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
        
        #pdb.set_trace()            
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance.append([acc, precision, sensitivity, specificity, MCC])
    print 'mean performance'
    print np.mean(np.array(all_performance), axis=0)

def build_deep_classical_autoencoder(autoencoder, input_dim, hidden_dim, activation, weight_reg = None, activity_reg = None):
    encoder = containers.Sequential([Dense(input_dim, hidden_dim, activation=activation, W_regularizer=weight_reg, activity_regularizer=activity_reg),
                            Dense(hidden_dim, hidden_dim/2, activation=activation)])
    decoder = containers.Sequential([Dense(hidden_dim/2, hidden_dim, activation=activation), 
                                     Dense(hidden_dim, input_dim, activation=activation, W_regularizer=weight_reg, activity_regularizer=activity_reg)])
    autoencoder.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    return autoencoder


def deep_autoencoder(datatype = 'RPI-Pred'):
    X, labels, chem_fea = get_data(datatype)
    y, encoder = preprocess_labels(labels)
    
    num_cross_val = 5
    batch_size  = 50
    nb_epoch = 100
    all_performance = []
    activation = 'linear' #'linear' #'relu, softmax, tanh'
    for fold in range(num_cross_val):
        train = []
        test = []
        train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
        test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        chem_train = np.array([x for i, x in enumerate(chem_fea) if i % num_cross_val != fold])
        chem_test = np.array([x for i, x in enumerate(chem_fea) if i % num_cross_val == fold])
                
        autoencoder = Sequential()
        autoencoder = build_deep_classical_autoencoder(autoencoder, train.shape[1], 256, activation)
        autoencoder.get_config(verbose=0)
        #norm_m0 = normalization.BatchNormalization((599,))
        #autoencoder.add(norm_m0)
        autoencoder.add(Dropout(0.5))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        autoencoder.compile(loss='mean_squared_error', optimizer= sgd) #'adam')
        # Do NOT use validation data with return output_reconstruction=True
        autoencoder.fit(train, train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0)
    
        prefilter_train = autoencoder.predict(train, verbose=0)
        prefilter_test = autoencoder.predict(test, verbose=0)
        #pdb.set_trace()
        '''
        #print prefilter_train.shape
        autoencoder2 = Sequential()
        autoencoder2 = build_deep_classical_autoencoder(autoencoder2, prefilter_train.shape[1], 128, activation)
        autoencoder2.get_config(verbose=0)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        autoencoder2.compile(loss='mean_squared_error', optimizer= sgd) #'adam')
        # Do NOT use validation data with return output_reconstruction=True
        autoencoder2.fit(prefilter_train, prefilter_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0)
    
        prefilter_train1 = autoencoder2.predict(prefilter_train, verbose=0)
        prefilter_test1 = autoencoder2.predict(prefilter_test, verbose=0)        
        '''
        prefilter_train = np.concatenate((prefilter_train, chem_train), axis = 1)
        prefilter_test = np.concatenate((prefilter_test, chem_test), axis = 1)
        
        print 'using random forest'
        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        #parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 2, 3, 4, 5, 6, 10], 'gamma': [0.5,1,2,4, 6, 8]}
        #svr = svm.SVC(probability = True)
        #clf = grid_search.GridSearchCV(svr, parameters, cv=3)        
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(prefilter_train, train_label_new)
        y_pred = clf.predict(prefilter_test)
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance.append([acc, precision, sensitivity, specificity, MCC])

        prefilter_train = []
        prefilter_test = []    
        '''
        print("Building classical fully connected layer for classification")
        model = Sequential()
        model.add(Dense(prefilter_train.shape[1], train_label.shape[1], activation=activation))
        model.add(Activation('softmax'))
    
        model.get_config(verbose=1)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(prefilter_train, train_label, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0, validation_data=(prefilter_test, test_label))
    
        score = model.evaluate(prefilter_test, test_label, verbose=0, show_accuracy=True)
        print('\nscore:', score) 
        '''  
    print 'mean performance'
    print np.mean(np.array(all_performance), axis=0)

def construct_one_layer_network(X_train, X_test, input_dim, output_dim, activation = 'linear', batch_size = 100, nb_epoch = 100):
    print 'constructing one-layer network'
    autoencoder = Sequential()
    autoencoder = build_deep_classical_autoencoder(autoencoder, input_dim, output_dim, activation)
    autoencoder.get_config(verbose=0)
    #norm_m0 = normalization.BatchNormalization((599,))
    #autoencoder.add(norm_m0)
    autoencoder.add(Dropout(0.5))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(loss='mean_squared_error', optimizer= sgd) #'adam')
    # Do NOT use validation data with return output_reconstruction=True
    autoencoder.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0)
    output_reconstruction=False
    first_train = autoencoder.predict(X_train, verbose=0)
    first_test = autoencoder.predict(X_test, verbose=0)
    
    return autoencoder, first_train, first_test

def multiple_layer_autoencoder(X_train, X_test, activation = 'linear', batch_size = 100, nb_epoch = 100):
    nb_hidden_layers = [X_train.shape[1], 256, 128]
    X_train_tmp = np.copy(X_train)
    X_test_tmp = np.copy(X_test)
    encoders = []
    for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
        print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
        # Create AE and training
        ae = Sequential()
        encoder = containers.Sequential([Dense(n_in, n_out, activation=activation)])
        decoder = containers.Sequential([Dense(n_out, n_in, activation=activation)])
        ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                           output_reconstruction=False))
        ae.add(Dropout(0.3))
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        ae.compile(loss='mean_squared_error', optimizer='rmsprop')
        ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=False, verbose=0)
        # Store trainined weight and update training data
        encoders.append(ae.layers[0].encoder)
        X_train_tmp = ae.predict(X_train_tmp)
        print X_train_tmp.shape
        X_test_tmp = ae.predict(X_test_tmp)
        
    return encoders, X_train_tmp, X_test_tmp
    '''model = Sequential()
    model.add(ae1[0].encoder)
    model.add(ae2[0].encoder)
    model.add(ae3[0].encoder)
    model.add(Dense(200, 10))
    model.add(Activation('softmax'))
    
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    '''
def get_preds( score1, score2, score3, weights):
    new_score  = [weights[0]*val1 + weights[1]* val2 + weights[2]* val3  for val1, val2, val3 in zip(score1, score2, score3)]
    return new_score

def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in proba]
    return label

def get_blend_data(j, clf, skf, X_test, X_dev, Y_dev, blend_train, blend_test):
        print 'Training classifier [%s]' % (j)
        blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print 'Fold [%s]' % (i)
            
            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]
            
            clf.fit(X_train, Y_train)
            
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = clf.predict_proba(X_cv)[:,1]
            blend_test_j[:, i] = clf.predict_proba(X_test)[:,1]
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)
    
        print 'Y_dev.shape = %s' % (Y_dev.shape)

def multiple_autoencoder_extract_feature(datatype = 'RPI-Pred'):
    X, labels, chem_fea = get_data(datatype)
    y, encoder = preprocess_labels(labels)
    num_cross_val = 5
    batch_size  = 50
    nb_epoch = 100
    all_performance = []
    all_performance_rf = []
    all_performance_ensemb = []
    all_performance_ae = []
    all_performance_rf_seq = []
    all_performance_chem = []
    all_performance_blend = []
    activation = 'linear' #'linear' #'relu, softmax, tanh'
    for fold in range(num_cross_val):
        train = []
        test = []
        train = np.array([x for i, x in enumerate(X) if i % num_cross_val != fold])
        test = np.array([x for i, x in enumerate(X) if i % num_cross_val == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % num_cross_val != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % num_cross_val == fold])
        chem_train = np.array([x for i, x in enumerate(chem_fea) if i % num_cross_val != fold])
        chem_test = np.array([x for i, x in enumerate(chem_fea) if i % num_cross_val == fold])
        
        '''
        print 'only use deep autoencoder'
        model = Sequential()
        for encoder in encoders:
            model.add(encoder)
        model.add(Dense(128, train_label.shape[1], activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        model.fit(train, train_label, nb_epoch=100, batch_size=100, verbose=0 )#, validation_split=0.15)
        ae_proba = model.predict_proba(test)[:,1]
        #pdb.set_trace()
        ae_y_pred = transfer_label_from_prob(ae_proba)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), ae_y_pred,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_ae.append([acc, precision, sensitivity, specificity, MCC])
        '''  
    
          
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        blend_train = np.zeros((train.shape[0], 5)) # Number of training data x Number of classifiers
        blend_test = np.zeros((test.shape[0], 5)) # Number of testing data x Number of classifiers 
        skf = list(StratifiedKFold(train_label_new, 5))  
                                
        encoders, prefilter_train, prefilter_test = multiple_layer_autoencoder(train, test, activation = 'linear', batch_size = 100, nb_epoch = 100)
     
        #prefilter_train = np.concatenate((prefilter_train, chem_train), axis = 1)
        #prefilter_test = np.concatenate((prefilter_test, chem_test), axis = 1)        
        print 'using random forest after sequence autoencoder'

        #parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 2, 3, 4, 5, 6, 10], 'gamma': [0.5,1,2,4, 6, 8]}
        #svr = svm.SVC(probability = True)
        #clf = grid_search.GridSearchCV(svr, parameters, cv=3)        
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(prefilter_train, train_label_new)
        y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        #pdb.set_trace()
        y_pred = transfer_label_from_prob(y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance.append([acc, precision, sensitivity, specificity, MCC])
        print '---' * 50
        get_blend_data(0, RandomForestClassifier(n_estimators=50), skf, prefilter_test, prefilter_train, np.array(train_label_new), blend_train, blend_test)
        
        prefilter_train = []
        prefilter_test = []
        print 'using random forest after chem feature autoencoder'
        encoders, prefilter_train, prefilter_test = multiple_layer_autoencoder(chem_train, chem_test, activation = 'linear', batch_size = 100, nb_epoch = 100)
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(prefilter_train, train_label_new)
        ae_y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        #pdb.set_trace()
        ae_y_pred = transfer_label_from_prob(ae_y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), ae_y_pred,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_ae.append([acc, precision, sensitivity, specificity, MCC])
        print '---' * 50
        get_blend_data(1, RandomForestClassifier(n_estimators=50), skf, prefilter_test, prefilter_train, np.array(train_label_new), blend_train, blend_test)
        
        prefilter_train = []
        prefilter_test = []
        tempary_train = np.concatenate((train, chem_train), axis = 1)
        tempary_test = np.concatenate((test, chem_test), axis = 1)    
        print 'using random forest after seqeunce and chem autoencoder'
        encoders, prefilter_train, prefilter_test = multiple_layer_autoencoder(tempary_train, tempary_test, activation = 'linear', batch_size = 100, nb_epoch = 100)
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(prefilter_train, train_label_new)
        all_y_pred_prob = clf.predict_proba(prefilter_test)[:,1]
        #pdb.set_trace()
        all_y_pred = transfer_label_from_prob(all_y_pred_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), all_y_pred,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_chem.append([acc, precision, sensitivity, specificity, MCC])
        print '---' * 50        
        get_blend_data(2, RandomForestClassifier(n_estimators=50), skf, prefilter_test, prefilter_train, np.array(train_label_new), blend_train, blend_test)
        
        print 'using RF using only chem feature'
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(chem_train, train_label_new)
        y_pred_rf_prob = clf.predict_proba(chem_test)[:,1]
        y_pred_rf = transfer_label_from_prob(y_pred_rf_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_rf,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_rf.append([acc, precision, sensitivity, specificity, MCC]) 
        print '---' * 50
        get_blend_data(3, RandomForestClassifier(n_estimators=50), skf, chem_test, chem_train, np.array(train_label_new), blend_train, blend_test)
        
        print 'using RF using only sequence feature'
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(train, train_label_new)
        y_pred_rf_prob = clf.predict_proba(test)[:,1]
        y_pred_rf = transfer_label_from_prob(y_pred_rf_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred_rf,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_rf_seq.append([acc, precision, sensitivity, specificity, MCC]) 
        print '---' * 50
        get_blend_data(4, RandomForestClassifier(n_estimators=50), skf, test, train, np.array(train_label_new), blend_train, blend_test)
        
        # Start blending!
        bclf = LogisticRegression()
        bclf.fit(blend_train, train_label_new)
        Y_test_predict = bclf.predict(blend_test)
        print 'blend result'
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), Y_test_predict,  real_labels)
        print acc, precision, sensitivity, specificity, MCC   
        all_performance_blend.append([acc, precision, sensitivity, specificity, MCC])     
        print '---' * 50
        '''
        print 'ensemble deep learning and rf'
        ensemb_prob = get_preds( y_pred_prob, y_pred_rf_prob, ae_y_pred, [0.3, 0.30, 0.4])       
        ensemb_label = transfer_label_from_prob(ensemb_prob)
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), ensemb_label,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
        all_performance_ensemb.append([acc, precision, sensitivity, specificity, MCC]) 
        print '---' * 50
        '''
    print 'in summary'
    print 'mean performance of chem autoencoder'
    print np.mean(np.array(all_performance_ae), axis=0)  
    print '---' * 50
    print 'mean performance of sequence autoencoder'
    print np.mean(np.array(all_performance), axis=0)
    print '---' * 50   
    print 'mean performance of only chem using RF'
    print np.mean(np.array(all_performance_rf), axis=0)
    print '---' * 50   
    print 'mean performance of sequence and chem autoencoder'
    print np.mean(np.array(all_performance_chem), axis=0) 
    print '---' * 50  
    print 'mean performance of only sequence with RF'
    print np.mean(np.array(all_performance_rf_seq), axis=0) 
    print '---' * 50      
    print 'mean performance of blend fusion'
    print np.mean(np.array(all_performance_blend), axis=0) 
    print '---' * 50 
           
def random_forest_classify(datatype = 'RPI-Pred'):
    X, labels, chem_fea = get_data(datatype)
    #X, scaler = preprocess_data(chem_fea)
    #y, encoder = preprocess_labels(labels)
    y = np.array(labels)
    #X_train, X_test, b_train, b_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #X_test, ids = load_data('test.csv', train=False)
    #X_test, _ = preprocess_data(X_test, scaler)
    
    dims = X.shape[1]
    print(dims, 'dims')
    num_cross_val = 5
    all_performance = []
    for fold in range(num_cross_val):
        train = []
        test = []
        train = [x for i, x in enumerate(X) if i % num_cross_val != fold]
        test = [x for i, x in enumerate(X) if i % num_cross_val == fold]
        train_label = [x for i, x in enumerate(y) if i % num_cross_val != fold]
        test_label = [x for i, x in enumerate(y) if i % num_cross_val == fold]
        print("Building model...")        
        print 'using random forest'
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(np.array(train), train_label)
        y_pred = clf.predict(np.array(test))
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(test_label), y_pred,  test_label)
        #print acc, precision, sensitivity, specificity, MCC
        all_performance.append([acc, precision, sensitivity, specificity, MCC])
    print 'mean performance'
    print np.mean(np.array(all_performance), axis=0)

def indep_validation():
    X, labels = prepare_feature(graph = False)
    print X.shape
    test, test_label = prepare_NPinter_feature(extract_only_posi = False, graph = False)
    print test.shape
    X, scaler = preprocess_data(X)
    test, _ = preprocess_data(test, scaler=scaler)
    print 'using random forest'
    clf = RandomForestClassifier(n_estimators=50)
    #calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    #calibrated_clf.fit(np.array(X), np.array(labels))
    #y_pred = calibrated_clf.predict(test)
    
    clf.fit(np.array(X), np.array(labels))
    #pdb.set_trace()
    y_pred = clf.predict(test)
    #acc, sensitivity = calculate_performace_without_MCC(len(test_label), y_pred,  test_label)
    #print acc, sensitivity  
    acc, precision, sensitivity, specificity, MCC = calculate_performace(len(test_label), y_pred,  test_label)
    print acc, precision, sensitivity, specificity, MCC  
#get_all_PDB_id()
#get_RNA_protein()
#get_NPinter_interaction()
def combine_deeplearning_randomforest(datatype = 'RPI-Pred'):
    X, labels, chem_fea = get_data(datatype)

datatype = sys.argv[1]
#random_forest_classify(datatype)
#deep_classifier_keras(datatype)
#deep_autoencoder(datatype)
multiple_autoencoder_extract_feature(datatype)

#get_own_lncRNA_protein()

#get_protein_trids()
#generate_negative_samples_RPI2241_RPI369('ncRNA-protein/RPI2241.fa', 'ncRNA-protein/RPI2241.txt', 'ncRNA-protein/RPI2241_all.txt')
#generate_negative_samples_RPI2241_RPI369('ncRNA-protein/RPI369.fa', 'ncRNA-protein/RPI369.txt', 'ncRNA-protein/RPI369_all.txt')
#get_RPI2241_RPI369_seq()
#get_RPI2241_RPI369_ind_file()
#indep_validation()
#read_protein_feature()       
#ncRNA_seq_dict = get_noncode_seq()