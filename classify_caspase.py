import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
#from sklearn.metrics import confusion_matrix

from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import gzip
from random import randint

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
#from sknn.mlp import Classifier, Layer
import pdb
import os

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
    
def get_all_PDB_id():
    protein_set =  get_uniq_pdb_protein_rna()
    fw = open('ncRNA-protein/all_seq.fa', 'w')
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
            name = line[1:] #discarding the initial >
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
            name_list.append(name)
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
    
    
def read_RNA_graph_feature(name_list, graph_file='ncRNA-protein/RNA_seq.gz.feature'):
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
        data[name_list[index]] = tmp_data
        index = index + 1
    fp.close()
    return data   

def read_protein_feature(protein_fea_file = 'ncRNA-protein/trainingSetFeatures.csv'):
    print protein_fea_file
    feature_dict = {}
    df = pd.read_csv(protein_fea_file)
    X = df.values.copy()
    for val in X:
        feature_dict[val[0]] = val[2:].tolist()
    
    #pdb.set_trace()
    return feature_dict

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
            fw_pro.write(ncRNA_dict[val] + '\n')
        else:
            print val
    fw_pro.close()     
    
    #return RNA_set, protein_set
def prepare_NPinter_feature():
    print 'NPinter data'
    protein_fea_dict = read_protein_feature(protein_fea_file ='ncRNA-protein/NPinter_trainingSetFeatures.csv') 
    name_list = read_name_from_fasta('ncRNA-protein/NPinter_RNA_seq.fa')
    RNA_fea_dict = read_RNA_graph_feature(name_list, graph_file='ncRNA-protein/npinter_RNA_seq.gz.feature')
    #pdb.set_trace()
    train = []
    label = []
    posi_set = set()
    pro_set = set()
    with open('ncRNA-protein/NPInter10412_dataset.txt', 'r') as fp:
        head  = True
        for line in fp:
            if head:
                head = False
                continue
            RNA, RNA_len, protein, protein_len, org = line.rstrip().split('\t')
            posi_set.add((RNA, protein))
            pro_set.add(protein)
            if RNA_fea_dict.has_key(RNA) and protein_fea_dict.has_key(protein):
                label.append(1)
                tmp_fea = RNA_fea_dict[RNA] + protein_fea_dict[protein]
                train.append(tmp_fea)
            else:
                print RNA, protein
    
    
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
                tmp_fea = RNA_fea_dict[RNA] + protein_fea_dict[select_pro]
                train.append(tmp_fea)
            else:
                print RNA, protein    
    #for key, val in RNA_fea_dict.iteritems():
        
            
    return np.array(train), label

def prepare_feature():
    print 'RPI-Pred data'
    name_list = read_name_from_fasta('ncRNA-protein/RNA_seq.fa')
    RNA_fea_dict = read_RNA_graph_feature(name_list) 
    #
    protein_fea_dict = read_protein_feature() 
    #pdb.set_trace()
    train = []
    label = []
    with open('ncRNA-protein/PositivePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            
            if protein_fea_dict.has_key(pro1) and RNA_fea_dict.has_key(pro2):
                label.append(1)
                tmp_fea = protein_fea_dict[pro1] + RNA_fea_dict[pro2]
                train.append(tmp_fea)
            else:
                print pro1, pro2
    with open('ncRNA-protein/NegativePairs.csv', 'r') as fp:
        for line in fp:
            if 'Protein ID' in line:
                continue
            pro1, pro2 = line.rstrip().split('\t')
            
            if protein_fea_dict.has_key(pro1) and RNA_fea_dict.has_key(pro2):
                label.append(0)
                tmp_fea = protein_fea_dict[pro1] + RNA_fea_dict[pro2]
                train.append(tmp_fea)
            else:
                print pro1, pro2

    return np.array(train), label



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

#def deep_classifier_keras(X, y, X_test):
def deep_classifier_keras():
    X, labels = prepare_NPinter_feature()
    #X, labels = prepare_feature() # load_data('train.csv', train=True)
    #X = pca_reduce_dimension(X)
    print X.shape
    X, scaler = preprocess_data(X)
    y, encoder = preprocess_labels(labels)
    
    #X_train, X_test, b_train, b_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #X_test, ids = load_data('test.csv', train=False)
    #X_test, _ = preprocess_data(X_test, scaler)
    nb_classes = y.shape[1]
    print(nb_classes, 'classes')
    #pdb.set_trace()
    
    dims = X.shape[1]
    print(dims, 'dims')
    num_cross_val = 10
    for fold in range(num_cross_val):
        train = []
        test = []
        train = [x for i, x in enumerate(X) if i % num_cross_val != fold]
        test = [x for i, x in enumerate(X) if i % num_cross_val == fold]
        train_label = [x for i, x in enumerate(y) if i % num_cross_val != fold]
        test_label = [x for i, x in enumerate(y) if i % num_cross_val == fold]
        print("Building model...")
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
        
        print("Training model...")
        
        model.fit(np.array(train), np.array(train_label), nb_epoch=20, batch_size=128, validation_split=0.15)
        weights = model.get_weights()
        #pdb.set_trace()
        print("Generating submission...")
        #pdb.set_trace()
        proba = model.predict_classes(np.array(test))
        '''
        #for pred, real in zip(proba, test_label):
        #    if real[0] == 1 and pred == 0:
        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)
        #pdb.set_trace()            
        #acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), proba,  real_labels)
        #print acc, precision, sensitivity, specificity, MCC
        
        print 'using random forest'
        clf = RandomForestClassifier(n_estimators=10)
        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)
        clf.fit(np.array(train), train_label_new)
        y_pred = clf.predict(np.array(test))
        acc, precision, sensitivity, specificity, MCC = calculate_performace(len(real_labels), y_pred,  real_labels)
        print acc, precision, sensitivity, specificity, MCC
#get_all_PDB_id()
#get_RNA_protein()
#get_NPinter_interaction()  
deep_classifier_keras()
#read_protein_feature()       
#ncRNA_seq_dict = get_noncode_seq()