import sys
import random
import ushuffle
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import pickle
from collections import OrderedDict

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
import ushuffle

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

   
def get_protein_motif_fig(filter_weights, filter_outs, out_dir, protein, seq_targets, sample_i = 0, structure = None):
    print 'plot motif fig', out_dir
    #seqs, seq_targets = get_seq_targets(protein)
    seqs = structure
    if sample_i:
        print 'sampling'
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)
            
        
        seq_targets = seq_targets[sample_i]
        filter_outs = filter_outs[sample_i]
    
    num_filters = filter_weights.shape[0]
    filter_size = 7

    
    #################################################################
    # individual filter plots
    #################################################################
    # also save information contents
    filters_ic = []
    meme_out = structure_motifs.meme_intro('%s/filters_meme.txt'%out_dir, seqs)

    for f in range(num_filters):
        print 'Filter %d' % f

        # plot filter parameters as a heatmap
        structure_motifs.plot_filter_heat(filter_weights[f,:,:], '%s/filter%d_heat.pdf' % (out_dir,f))

        # write possum motif file
        structure_motifs.filter_possum(filter_weights[f,:,:], 'filter%d'%f, '%s/filter%d_possum.txt'%(out_dir,f), False)
        
        structure_motifs.plot_filter_logo(filter_outs[:,:, f], filter_size, seqs, '%s/filter%d_logo'%(out_dir,f), maxpct_t=0.5)
        
        filter_pwm, nsites = structure_motifs.make_filter_pwm('%s/filter%d_logo.fa'%(out_dir,f))
        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            structure_motifs.meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()
    
            
def get_motif_fig(filter_weights, filter_outs, out_dir, protein, sample_i = 0):
    print 'plot motif fig', out_dir
    seqs, seq_targets = get_seq_targets(protein)
    if sample_i:
        print 'sampling'
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)
            
        
        seq_targets = seq_targets[sample_i]
        filter_outs = filter_outs[sample_i]
    
    num_filters = filter_weights.shape[0]
    filter_size = 7

    #pdb.set_trace()
    #################################################################
    # individual filter plots
    #################################################################
    # also save information contents
    filters_ic = []
    meme_out = meme_intro('%s/filters_meme.txt'%out_dir, seqs)

    for f in range(num_filters):
        print 'Filter %d' % f

        # plot filter parameters as a heatmap
        plot_filter_heat(filter_weights[f,:,:], '%s/filter%d_heat.pdf' % (out_dir,f))

        # write possum motif file
        filter_possum(filter_weights[f,:,:], 'filter%d'%f, '%s/filter%d_possum.txt'%(out_dir,f), False)

        # plot weblogo of high scoring outputs
        plot_filter_logo(filter_outs[:,:, f], filter_size, seqs, '%s/filter%d_logo'%(out_dir,f), maxpct_t=0.5)

        # make a PWM for the filter
        filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa'%(out_dir,f))

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()


    #################################################################
    # annotate filters
    #################################################################
    # run tomtom #-evalue 0.01 
    subprocess.call('tomtom -dist pearson -thresh 0.05 -eps -oc %s/tomtom %s/filters_meme.txt %s' % (out_dir, out_dir, 'Ray2013_rbp_RNA.meme'), shell=True)

    # read in annotations
    filter_names = name_filters(num_filters, '%s/tomtom/tomtom.txt'%out_dir, 'Ray2013_rbp_RNA.meme')


    #################################################################
    # print a table of information
    #################################################################
    table_out = open('%s/table.txt'%out_dir, 'w')

    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
    print >> table_out, '%3s  %19s  %10s  %5s  %6s  %6s' % header_cols

    for f in range(num_filters):
        # collapse to a consensus motif
        consensus = filter_motif(filter_weights[f,:,:])

        # grab annotation
        annotation = '.'
        name_pieces = filter_names[f].split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[1]

        # plot density of filter output scores
        fmean, fstd = plot_score_density(np.ravel(filter_outs[:,:, f]), '%s/filter%d_dens.pdf' % (out_dir,f))

        row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
        print >> table_out, '%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols

    table_out.close()


    #################################################################
    # global filter plots
    #################################################################
    if True:
        new_outs = []
        for val in filter_outs:
            new_outs.append(val.T)
        filter_outs = np.array(new_outs)
        print filter_outs.shape
        # plot filter-sequence heatmap
        plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf'%out_dir)


def get_RNA_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def get_protein_concolutional_array(seq, motif_len = 7):       
    alpha = 'ABCDEFG'
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 7))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.14]*7)
    
    for i in range(row-6, row):
        new_array[i] = np.array([0.14]*7)

    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in alpha:
            new_array[i] = np.array([0.14]*7)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        
    return new_array



def get_feature(model, X_batch, index=0):
    inputs = [K.learning_phase()] + [model.inputs[index]]
    _convout1_f = K.function(inputs, model.layers[0].layers[index].layers[1].output)
    activations =  _convout1_f([0] + [X_batch[index]])
    
    return activations

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
   
def get_motif(model, testing, protein, y, index = 0, dir1 = 'seq_cnn/', structure  = None):
    sfilter = model.layers[0].layers[index].layers[0].get_weights()
    filter_weights_old = np.transpose(sfilter[0][:,0,:,:], (2, 1, 0)) #sfilter[0][:,0,:,:]
    print filter_weights_old.shape
    #pdb.set_trace()
    filter_weights = []
    for x in filter_weights_old:
        #normalized, scale = preprocess_data(x)
        #normalized = normalized.T
        #normalized = normalized/normalized.sum(axis=1)[:,None]
        x = x - x.mean(axis = 0)
        filter_weights.append(x)
        
    filter_weights = np.array(filter_weights)
    #pdb.set_trace()
    filter_outs = get_feature(model, testing, index)
    #pdb.set_trace()
    
    #sample_i = np.array(random.sample(xrange(testing.shape[0]), 500))
    sample_i =0

    out_dir = dir1 + protein
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if index == 0:    
        get_motif_fig(filter_weights, filter_outs, out_dir, protein, sample_i)
    else:
        get_protein_motif_fig(filter_weights, filter_outs, out_dir, protein, y, sample_i, structure)

def set_cnn_model_old(input_dim, input_length):
    nbfilter = 102
    model = Sequential()
    #model.add(brnn)
    model.add(Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=10,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))

    model.add(Dropout(0.5))

    return model

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
    model.add(Convolution1D(nb_filter, filter_length, border_mode='same', init='glorot_normal'))
    model.add(LeakyReLU(.3))
    model.add(MaxPooling1D(pool_length=3))
    model.add(Dropout(dropout))
    
    #model.add(Convolution1D(nb_filter, filter_length, border_mode='same', init='glorot_normal'))
    #model.add(LeakyReLU(.3))
    #model.add(Dropout(dropout))
    
    return model

def get_cnn_network_graphprot(rna_len = 501, nb_filter = 16):
    #nbfilter = 32
    print 'configure cnn network'
    #embedded_dim, embedding_weights, n_aa_symbols = get_embed_dim('peptideEmbedding.pickle')
    #seq_model = set_cnn_embed(n_aa_symbols, pro_len, embedded_dim, embedding_weights)
    #print n_aa_symbols
    embedded_rna_dim, embedding_rna_weights, n_nucl_symbols = get_embed_dim('rnaEmbedding.pickle')
    print 'symbol', n_nucl_symbols
    model = set_cnn_embed(n_nucl_symbols, rna_len, embedded_rna_dim, embedding_rna_weights, nb_filter = nb_filter)
    #pdb.set_trace()
    #print 'pro cnn', seq_model.output_shape
    #print 'rna cnn', struct_model.output_shape
    #model = Sequential()
    #model.add(Merge([seq_model, struct_model], mode='concat', concat_axis=1))
    
    #model.add(Bidirectional(LSTM(2*nbfilter)))
    #model.add(Dropout(0.10))
    model.add(Flatten())
    #model.add(Dense(nbfilter*(n_aa_symbols + n_nucl_symbols), activation='relu'))
    #model.add(Dropout(0.50))
    model.add(Dense(nb_filter*50, activation='relu')) 
    model.add(Dropout(0.50))
    model.add(Dense(nb_filter*10, activation='relu')) 
    model.add(Dropout(0.50))
    #model.add(BatchNormalization(mode=2))
    print model.output_shape
    
    return model

def get_cnn_network_embed(rna_len, pro_len):
    nbfilter = 32
    print 'configure cnn network'
    embedded_dim, embedding_weights, n_aa_symbols = get_embed_dim('peptideEmbedding.pickle')
    seq_model = set_cnn_embed(n_aa_symbols, pro_len, embedded_dim, embedding_weights)
    print n_aa_symbols
    embedded_rna_dim, embedding_rna_weights, n_nucl_symbols = get_embed_dim('rnaEmbedding.pickle')
    print 'symbol', n_nucl_symbols
    struct_model = set_cnn_embed(n_nucl_symbols, rna_len, embedded_rna_dim, embedding_rna_weights)
    #pdb.set_trace()
    print 'pro cnn', seq_model.output_shape
    print 'rna cnn', struct_model.output_shape
    model = Sequential()
    model.add(Merge([seq_model, struct_model], mode='concat', concat_axis=1))
    
    #model.add(Bidirectional(LSTM(2*nbfilter)))
    #model.add(Dropout(0.10))
    model.add(Flatten())
    #model.add(Dense(nbfilter*(n_aa_symbols + n_nucl_symbols), activation='relu'))
    #model.add(Dropout(0.50))
    model.add(Dense(nbfilter*100, activation='relu')) 
    model.add(Dropout(0.50))
    #model.add(BatchNormalization(mode=2))
    print model.output_shape
    
    return model
        
def run_network(model, total_hid, training, testing, y, validation, val_y):
    model.add(Dense(2, input_shape=(total_hid,)))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)#'rmsprop')
    #pdb.set_trace()
    print 'model training'
    #checkpointer = ModelCheckpoint(filepath="models/bestmodel.hdf5", verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(training, y, batch_size=100, nb_epoch=10, verbose=0, validation_data=(validation, val_y), callbacks=[earlystopper])
    
    #pdb.set_trace()
    #get_motif(model, testing, protein, y, index = 0, dir1 = 'seq_cnn/')
    #get_motif(model, testing, protein, y, index = 1, dir1 = 'structure_cnn/', structure = structure)
    #new_out = get_feature(model, testing)
    #pdb.set_trace()
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

aa_dict = OrderedDict([
               ('A', 1), 
               ('C', 2),
               ('E', 3),
               ('D', 4),
               ('G', 5),
               ('F', 6),
               ('I', 7),
               ('H', 8),
               ('K', 9),
               ('M', 10),
               ('L', 11),
               ('N', 12),
               ('Q', 13),
               ('P', 14),
               ('S', 15),
               ('R', 16),
               ('T', 17),
               ('W', 18),
               ('V', 19),
               ('Y', 20)
               ])

def read_rna_dict():
    odr_dict = {}
    with open('rna_nul_dict', 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict
                

def aa_integerMapping(peptideSeq):
    peptideArray = []
    for aa in peptideSeq:
        if aa_dict.has_key(aa):
            peptideArray.append(aa_dict[aa])
        else:
            peptideArray.append(-1)

    return np.asarray(peptideArray)

#featureMatrix = np.empty((0, peptide_n_mer), int)
#for num in range(len(seqMatrix)):
#  featureMatrix = np.append(featureMatrix, [aa_integerMapping(seqMatrix.iloc[num])], axis=0)

  
def loaddata(datadir = 'data/', ushuffle = True):
    #pair_file = datadir + 'interactions_HT.txt'
    pair_file = datadir + 'test_part2'
    rbp_seq_file = datadir + 'rbps_HT.fa'
    rna_seq_file = datadir + 'utrs.fa'
    pairs = []
    rbp_seq_dict = read_fasta_file(rbp_seq_file)
    pro_len = [] #1101
    for val in rbp_seq_dict.values():
        pro_len.append(len(val))
    pro_len.sort()
    pro_nax_len = pro_len[int(len(pro_len)*0.9)]    
    rna_seq_dict = read_fasta_file(rna_seq_file)
    rna_len = [] # 2695
    for val in rna_seq_dict.values():
        rna_len.append(len(val))
    
    rna_len.sort()
    rna_nax_len = rna_len[int(len(rna_len)*0.9)] 
    #groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    #group_dict = TransDict_from_list(groups)
    #pdb.set_trace()
    label = []
    rna_array = []
    protein_array = []
    trids = get_6_trids()
    nn_dict = read_rna_dict()
    with open(pair_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split()
            protein = values[0]
            rna = values[1]
            protein_seq = rbp_seq_dict[protein]
            #protein_seq = translate_sequence (protein_seq, group_dict)
            protein_seq = padding_sequence(protein_seq, max_len = pro_nax_len, repkey = 'B')
            #protein_seq = list(protein_seq)
            #onehot_pro = get_protein_concolutional_array(protein_seq)
            protein_array.append(aa_integerMapping(protein_seq))
            #pdb.set_trace()
            rna_seq = rna_seq_dict[rna]
            rna_seq = rna_seq.replace('T', 'U')
            rna_seq_pad = padding_sequence(rna_seq, max_len = rna_nax_len, repkey = 'N')
            #onehot_rna = get_RNA_concolutional_array(rna_seq_pad)
            tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
            rna_array.append(tri_feature)
            label.append(1)
            if ushuffle:
                protein_array.append(aa_integerMapping(protein_seq))
                shuffle_rna_seq = local_ushuffle(rna_seq)
                shuffle_rna_seq_pad = padding_sequence(shuffle_rna_seq, max_len = rna_nax_len, repkey = 'N')
                #onehot_rna = get_RNA_concolutional_array(shuffle_rna_seq_pad)
                tri_feature_shu = get_6_nucleotide_composition(trids, shuffle_rna_seq_pad, nn_dict)
                rna_array.append(tri_feature_shu)
                #rna_array.append(list(shuffle_rna_seq_pad))
                label.append(0)
    
    return np.array(protein_array), np.array(rna_array), np.array(label), rna_nax_len, pro_nax_len

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
                '''
                seq_len = len(seq)
                if seq_len < min_len:
                    continue
                
                gap_ind = (seq_len- min_len)/2 
                new_seq = seq[gap_ind:len(seq) - gap_ind]
                if len(new_seq) > min_len:
                    new_seq = new_seq[:min_len]
                if len(new_seq) < min_len:
                    pdb.set_trace()
                '''
                #pdb.set_trace()
                #seq_array = get_RNA_seq_concolutional_array(new_seq)
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
            #strucures, labels = read_structure_graphprot(os.path.join(path, tmpfile), label = label, min_len = min_len)
            #mix_structure = mix_structure + strucures
    #tmp.append(np.array(mix_seq))
    #tmp.append(np.array(mix_structure))
    
    data["seq"] = mix_seq
    data["Y"] = np.array(mix_label)
    
    return data

def load_predict_graphprot_data():
    data_dir = '/home/panxy/eclipse/rna-protein/data/GraphProt_CLIP_sequences/'
    fw = open('result_file_graphprot_size_10', 'w')
    seq_hid = 16
    for protein in os.listdir(data_dir):

        protein = protein.split('.')[0]
        print protein
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

def loaddata_graphprot(protein, train = True, ushuffle = True):
    #pdb.set_trace()
    data = load_graphprot_data(protein, train = train)
    label = data["Y"]
    rna_array = []
    protein_array = []
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

def load_data():
    data = dict()
    tmp = []
    rna_array, protein_array, label, rna_nax_len, pro_nax_len = loaddata()
    #print rna_array.shape, protein_array.shape
    tmp.append(rna_array)
    #seq_onehot, structure = read_structure(os.path.join(path, 'sequences.fa.gz'), path)
    tmp.append(protein_array)
    data["seq"] = tmp
    #data["structure"] = structure
    
    
    data["Y"] = label
    
    return data, rna_nax_len, pro_nax_len

    
def calculate_auc(net, hid, train, test, true_y, train_y, validation = None, val_y = None, model_name = None):
    predict, model = run_network(net, hid, train, test, train_y, validation, val_y)
    #pdb.set_trace()
    auc = roc_auc_score(true_y, predict)
        
    print "Test AUC: ", auc
    #with open(model_name, 'w') as f:
    #    pickle.dump(model, f)
    #model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    #del model

    return auc, predict    


def run_RNA_protein(fw = None):
    data, rna_nax_len, pro_nax_len = load_data()
    rna_nax_len = rna_nax_len -5
    print len(data), rna_nax_len, pro_nax_len
    print 'finishing load data'
    seq_hid =16
    protein_hid = 16
    #pdb.set_trace()   
    
    '''
    seq_hid = 16
    struct_hid = 16
    #pdb.set_trace()
    train_Y = training_data["Y"]
    print len(train_Y)
    #pdb.set_trace()
    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y)
    '''
    #pdb.set_trace()
    training_indice, training_label, test_indice, test_label = split_training_validation(data["Y"])
    training_data = dict()
    test_data = dict()
    #for key in data.keys():
    training_data["seq"] = []
    test_data["seq"] = []
    training_data["seq"].append(data["seq"][0][training_indice])
    training_data["seq"].append(data["seq"][1][training_indice])
    test_data["seq"].append(data["seq"][0][test_indice])
    test_data["seq"].append(data["seq"][1][test_indice])
    training_data["Y"] = data["Y"][training_indice] 
    test_data["Y"] = data["Y"][test_indice]
    #true_y = test_data["Y"].copy()    
    training_indice, training_label, validation_indice, validation_label = split_training_validation(training_data["Y"])
    #pdb.set_trace()
    cnn_train  = []
    cnn_validation = []
    seq_data = training_data["seq"][0]
    print seq_data.shape
    #pdb.set_trace()
    seq_train = seq_data[training_indice]
    seq_validation = seq_data[validation_indice] 
    struct_data = training_data["seq"][1]
    print struct_data.shape
    struct_train = struct_data[training_indice]
    struct_validation = struct_data[validation_indice] 
    cnn_train.append(seq_train)
    cnn_train.append(struct_train)
    cnn_validation.append(seq_validation)
    cnn_validation.append(struct_validation)        
    seq_net =  get_cnn_network_embed(rna_nax_len, pro_nax_len)
    seq_data = []
            
    y, encoder = preprocess_labels(training_label)
    val_y, encoder = preprocess_labels(validation_label, encoder = encoder) 
    
    training_data.clear()

    #pdb.set_trace()
    
    true_y = test_data["Y"].copy()
    
    print 'predicting'    

    seq_test = test_data["seq"]
    seq_auc, seq_predict = calculate_auc(seq_net, seq_hid + protein_hid, cnn_train, seq_test, true_y, y, validation = cnn_validation,
                                          val_y = val_y)
    seq_train = []
    seq_test = []
         
        
        
    print str(seq_auc)
    fw.write( str(seq_auc) +'\n')

    mylabel = "\t".join(map(str, true_y))
    myprob = "\t".join(map(str, seq_predict))  
    fw.write(mylabel + '\n')
    fw.write(myprob + '\n')
    
if __name__ == "__main__":
    #get_all_rpbs()
    #run_RNA_protein()
    load_predict_graphprot_data()
    #loaddata()   
    #or_dict = read_rna_dict()
    #pdb.set_trace()
    
    