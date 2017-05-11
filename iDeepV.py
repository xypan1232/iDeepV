import sys
import random
import ushuffle
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import pickle
from collections import OrderedDict
import gzip
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import LSTM, Bidirectional 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from seq_motifs import *
import structure_motifs
#import ushuffle

def generate_sequence_with_same_componenet(sequence):   
    random.shuffle(sequence)

    return ''.join(sequence)

def do_dinucleotide_shuffling(seq):
    seq1 = ushuffle.shuffle(seq, len(seq), 2)
    return seq1

def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
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

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def split_training_validation(classes, validation_size = 0.2, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label        

def get_embed_dim(embed_file):
    with open(embed_file) as f:
        pepEmbedding = pickle.load(f)
        
    embedded_dim = pepEmbedding[0].shape
    print embedded_dim
    n_aa_symbols, embedded_dim = embedded_dim
    print n_aa_symbols, embedded_dim
    # = embedded_dim[0]
    embedding_weights = np.zeros((n_aa_symbols + 1,embedded_dim))
    embedding_weights[1:,:] = pepEmbedding[0]
    
    return embedded_dim, embedding_weights, n_aa_symbols


def set_cnn_embed(n_aa_symbols, input_length, embedded_dim, embedding_weights, nb_filter = 16):
    #nb_filter = 64
    filter_length = 10
    dropout = 0.5
    model = Sequential()
    #pdb.set_trace()
    model.add(Embedding(input_dim=n_aa_symbols+1, output_dim = embedded_dim, weights=[embedding_weights], input_length=input_length, trainable = True))
    print 'after embed', model.output_shape
    model.add(Convolution1D(nb_filter, filter_length, border_mode='valid', init='glorot_normal'))
    model.add(Activation(LeakyReLU(.3)))
    model.add(MaxPooling1D(pool_length=3))
    model.add(Dropout(dropout))
    
    return model

def get_cnn_network_graphprot(rna_len = 501, nb_filter = 16):
    print 'configure cnn network'
    embedded_rna_dim, embedding_rna_weights, n_nucl_symbols = get_embed_dim('rnaEmbedding25.pickle')
    print 'symbol', n_nucl_symbols
    model = set_cnn_embed(n_nucl_symbols, rna_len, embedded_rna_dim, embedding_rna_weights, nb_filter = nb_filter)
    
    #model.add(Bidirectional(LSTM(2*nbfilter)))
    #model.add(Dropout(0.10))
    model.add(Flatten())
    model.add(Dense(nb_filter*50, activation='relu')) 
    model.add(Dropout(0.50))
    model.add(Dense(nb_filter*10, activation='sigmoid')) 
    model.add(Dropout(0.50))
    print model.output_shape
    
    return model

        
def run_network(model, total_hid, training, testing, y, validation, val_y):
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)sgd)#'
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #pdb.set_trace()
    print 'model training'
    #checkpointer = ModelCheckpoint(filepath="models/bestmodel.hdf5", verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(training, y, batch_size=100, nb_epoch=10, verbose=0, validation_data=(validation, val_y), callbacks=[earlystopper])
    
    predictions = model.predict_proba(testing)[:,1]
    return predictions, model

def padding_sequence(seq, max_len = 2695, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

def local_ushuffle(seq, dishuff = True):
    '''
    shuffle dinucletide
    '''

    if dishuff:
        return_seq = ushuffle.shuffle(seq, len(seq), 2)
    else:
        l = list(seq)
        random.shuffle(l)
        return_seq = ''.join(l)        
        
    return return_seq

def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        #let's discard the newline at the end (if any)
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[1:] #discarding the initial >
            seq_dict[name] = ''
        else:
            #it is sequence
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    
    return seq_dict

def read_fasta_file_uniprot(fasta_file):
    seq_dict = {}    
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        #let's discard the newline at the end (if any)
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            if '_HUMAN' in line:
                human  =True
                name = line[1:].split('|')[1] #discarding the initial >
                seq_dict[name] = ''
            else:
                human = False
        else:
            #it is sequence
            if human:
                seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    
    return seq_dict

def get_all_rpbs():
    fw = open('./data/allrbps.fa', 'w')
    seq_dict1 = read_fasta_file('./data/rbps_HT.fa')
    seq_dict2 = read_fasta_file('./data/rbps_new.fa')
    seq_dict3 = read_fasta_file('./data/star_rbps.fa')
    seq_dict4 = read_fasta_file_uniprot('./data/RBP2780.txt')
    seq_dict5 = read_fasta_file_uniprot('./data/Human-RBP967.txt')
    
    #pdb.set_trace()
    all_keys = set(seq_dict1.keys()) | set(seq_dict2.keys())| set(seq_dict3.keys())| set(seq_dict4.keys())| set(seq_dict5.keys())
    for key in all_keys:
        if seq_dict1.has_key(key):
            fw.write('>' + key + '\n')
            fw.write(seq_dict1[key] + '\n')
            continue
        if seq_dict2.has_key(key):
            fw.write('>' + key + '\n')
            fw.write(seq_dict2[key] + '\n')
            continue
        if seq_dict3.has_key(key):
            fw.write('>' + key + '\n')
            fw.write(seq_dict3[key] + '\n')
            continue
        if seq_dict4.has_key(key):
            fw.write('>' + key + '\n')
            fw.write(seq_dict4[key] + '\n')
            continue
        if seq_dict5.has_key(key):
            fw.write('>' + key + '\n')
            fw.write(seq_dict5[key] + '\n')
            continue
    fw.close()

def read_gencode_seq(fastafile = 'gencode.v19.lncRNA_transcripts.fa.gz'):
    seq_dict = {}
    with open(fastafile, 'r') as fp:
        for line in fp:
            line = line.rstrip()
            if line[0]=='>': #or line.startswith('>')
                name = line[1:].split('|')[5] #discarding the initial >
                seq_dict[name] = ''
            else:
                #it is sequence
                seq_dict[name] = seq_dict[name] + line.upper()
                
    return seq_dict

def get_random_pair():
    random_pair = []

def get_6_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**6
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        n=n/base
        ch3=chars[n%base]
        n=n/base
        ch4=chars[n%base]
        n=n/base
        ch5=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    return  nucle_com   

def get_6_nucleotide_composition(tris, seq, ordict):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    #tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer)
            tri_feature.append(ordict[str(ind)])
        else:
            tri_feature.append(-1)
    #tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()        
    return np.asarray(tri_feature)

def read_rna_dict():
    odr_dict = {}
    with open('rna_dict', 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict


def get_rna_seqs_rec(rnas, rna_seq_dict, rna_nax_len, trids, nn_dict):

    label = []
    rna_array = []
    for rna in rnas:
        rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        rna_seq_pad = padding_sequence(rna_seq, max_len = rna_nax_len, repkey = 'N')
        tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
        rna_array.append(tri_feature)
        label.append(1)
        if ushuffle:
            shuffle_rna_seq = local_ushuffle(rna_seq)
            shuffle_rna_seq_pad = padding_sequence(shuffle_rna_seq, max_len = rna_nax_len, repkey = 'N')
            #onehot_rna = get_RNA_concolutional_array(shuffle_rna_seq_pad)
            tri_feature_shu = get_6_nucleotide_composition(trids, shuffle_rna_seq_pad, nn_dict)
            label.append(0)
            rna_array.append(tri_feature_shu)
    
    return np.array(rna_array), np.array(label)
    
def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq_list.append(seq)
                labels.append(label)

    return seq_list, labels

def load_graphprot_data(protein, train = True, path = '/home/panxy/eclipse/rna-protein/data/GraphProt_CLIP_sequences/'):
    data = dict()
    tmp = []
    listfiles = os.listdir(path)
    
    key = '.train.'
    if not train:
        key = '.ls.'
    mix_label = []
    mix_seq = []
    mix_structure = []    
    for tmpfile in listfiles:
        if protein not in tmpfile:
            continue
        if key in tmpfile:
            if 'positive' in tmpfile:
                label = 1
            else:
                label = 0
            seqs, labels = read_seq_graphprot(os.path.join(path, tmpfile), label = label)
            #pdb.set_trace()
            mix_label = mix_label + labels
            mix_seq = mix_seq + seqs
    
    data["seq"] = mix_seq
    data["Y"] = np.array(mix_label)
    
    return data

def loaddata_graphprot(protein, train = True, ushuffle = True):
    #pdb.set_trace()
    data = load_graphprot_data(protein, train = train)
    label = data["Y"]
    rna_array = []
    trids = get_6_trids()
    nn_dict = read_rna_dict()
    for rna_seq in data["seq"]:
        #rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        rna_seq_pad = padding_sequence(rna_seq, max_len = 501, repkey = 'N')
        #onehot_rna = get_RNA_concolutional_array(rna_seq_pad)
        tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
        rna_array.append(tri_feature)
    
    return np.array(rna_array), label

def load_predict_graphprot_data():
    data_dir = '/home/panxy/eclipse/rna-protein/data/GraphProt_CLIP_sequences/'
    fw = open('result_file_graphprot_new', 'w')
    seq_hid = 16
    finished_protein = set()
    for protein in os.listdir(data_dir):
        if protein in finished_protein:
            continue
        protein = protein.split('.')[0]
        print protein
        finished_protein.add(protein)
        fw.write(protein + '\t')
        data, label = loaddata_graphprot(protein)
        seq_net = get_cnn_network_graphprot(rna_len = 496, nb_filter = seq_hid)
        #pdb.set_trace()
        #true_y = test_data["Y"].copy()
        training_indice, training_label, val_indice, val_label = split_training_validation(label)
        cnn_train = data[training_indice]
        training_label = label[training_indice]
        cnn_validation = data[val_indice]
        validation_label = label[val_indice]        
        y, encoder = preprocess_labels(training_label)
        val_y, encoder = preprocess_labels(validation_label, encoder = encoder) 
        print 'predicting'    
        test_data, true_y = loaddata_graphprot(protein, train = False)
        #seq_test = test_data["seq"]
        model_name = 'model/' + protein +'.pickle'
        seq_auc, seq_predict = calculate_auc(seq_net, seq_hid, cnn_train, test_data, true_y, y, validation = cnn_validation,
                                              val_y = val_y, model_name = model_name)

        print str(seq_auc)
        fw.write( str(seq_auc) +'\n')
        mylabel = "\t".join(map(str, true_y))
        myprob = "\t".join(map(str, seq_predict))  
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
         
    fw.close()

    
def calculate_auc(net, hid, train, test, true_y, train_y, validation = None, val_y = None, model_name = None):
    predict, model = run_network(net, hid, train, test, train_y, validation, val_y)
    #pdb.set_trace()
    auc = roc_auc_score(true_y, predict)
        
    print "Test AUC: ", auc

    return auc, predict 
   
def read_seq(seq_file, trids, nn_dict):
    seq_list = []
    label_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                posi_label = name.split(';')[-1]
                label = posi_label.split(':')[-1]
                label_list.append(int(label))
                if len(seq):
                    #seq_array = get_RNA_seq_concolutional_array(seq)
                    seq_list.append(seq)                    
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_list.append(seq) 
    
    rna_array = []
    for rna_seq in seq_list:
        rna_seq = rna_seq.replace('T', 'U')
        rna_seq_pad = padding_sequence(rna_seq, max_len = 101, repkey = 'N')
        tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
        rna_array.append(tri_feature)
           
    return np.array(rna_array), np.array(label_list)

def run_rbp31():
    data_dir = '/home/panxy/eclipse/ideep/iDeep/datasets/clip'
    rna_max_len = 101
    seq_hid = 16
    trids = get_6_trids()
    nn_dict = read_rna_dict()
    fw = open('result_file_rbp31', 'w')
    for protein in os.listdir(data_dir):
        print protein
        fw.write(protein + '\t')
        path =  "/home/panxy/eclipse/ideep/ideep/datasets/clip/%s/30000/training_sample_0" % protein
        data, label = read_seq(os.path.join(path, 'sequences.fa.gz'), trids, nn_dict)
        seq_net = get_cnn_network_graphprot(rna_len = rna_max_len - 5, nb_filter = seq_hid)
        
        training_indice, training_label, val_indice, val_label = split_training_validation(label)
        cnn_train = data[training_indice]
        training_label = label[training_indice]
        cnn_validation = data[val_indice]
        validation_label = label[val_indice]        
        y, encoder = preprocess_labels(training_label)
        val_y, encoder = preprocess_labels(validation_label, encoder = encoder) 
        
        print 'testing'    
        path =  "/home/panxy/eclipse/ideep/ideep/datasets/clip/%s/30000/test_sample_0" % protein
        test_data, true_y = read_seq(os.path.join(path, 'sequences.fa.gz'), trids, nn_dict)
        model_name = 'model/' + protein +'.pickle'
        seq_auc, seq_predict = calculate_auc(seq_net, seq_hid, cnn_train, test_data, true_y, y, validation = cnn_validation,
                                              val_y = val_y, model_name = model_name)

        print str(seq_auc)
        fw.write( str(seq_auc) +'\n')
        mylabel = "\t".join(map(str, true_y))
        myprob = "\t".join(map(str, seq_predict))  
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
         
    fw.close()        

def run_ideepv(dataset = "RBP-24"):
    if dataset == "RBP-24":
        load_predict_graphprot_data()
    if dataset == "RBP-31":
        run_rbp31()
    
if __name__ == "__main__":
    dataset = sys.argv[1]
    run_ideepv(dataset)
    
    
