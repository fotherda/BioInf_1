from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import datetime#, time
import pickle as pi

from BioInf.protein import *
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from BioInf.protein import *

from timeit import default_timer as timer

from tensorflow.examples.tutorials.mnist import input_data
from Advanced_1.convergenceTester import ConvergenceTester
from Advanced_1.learningRateScheduler import LearningRateScheduler
from Advanced_1.dataBatcher import DataBatcher
from scipy.stats import ttest_ind
from scipy.misc import toimage

NUM_CATEGORIES = 4

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)
  
def build_network_task1(x, nrecurrent_units, cell, y_, use_batch_norm):
    
    raw_rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    last_rnn_output = tf.slice(raw_rnn_outputs, [0, raw_rnn_outputs.get_shape()[1].value - 1, 0], 
                               [-1, 1, raw_rnn_outputs.get_shape()[2].value] )
    last_rnn_output = tf.squeeze(last_rnn_output, 1, name='sliced_rnn_outputs')

    W_2 = weight_variable([nrecurrent_units, 100])
    b_2 = bias_variable([100])   
    lin_1 = tf.matmul(last_rnn_output, W_2) + b_2
    
    if use_batch_norm:
        # Calculate batch mean and variance
        batch_mean1, batch_var1 = tf.nn.moments(lin_1,[0])
    
        # Apply the initial batch normalizing transform
        lin_1_hat = (lin_1 - batch_mean1) / tf.sqrt(batch_var1 + 1e-3)
        
        # Create two new parameters, scale and beta (shift)
        scale1 = tf.Variable(tf.ones([100]))
        beta1 = tf.Variable(tf.zeros([100]))
        
        # Scale and shift to obtain the final output of the batch normalization
        # this value is fed into the activation function (here a sigmoid)
        BN1 = scale1 * lin_1_hat + beta1
        h_2 = tf.nn.relu(BN1)    
    else:
        h_2 = tf.nn.relu(lin_1)
        
    W_3 = weight_variable([100, 10])
    b_3 = bias_variable([10])
    y = tf.matmul(h_2, W_3) + b_3
    
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
    return y, cross_entropy

def build_network_task2(x, nrecurrent_units, cell, y_):
    
    raw_rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32) # batch_size x 783 x 32
    
    W_1 = weight_variable([nrecurrent_units, 1])
    b_1 = bias_variable([1])   
         
    reshaped_outputs = tf.reshape(raw_rnn_outputs, [-1, nrecurrent_units])
    logits = tf.matmul(reshaped_outputs, W_1) + b_1
    logits = tf.reshape(logits, [-1, raw_rnn_outputs.get_shape()[1].value, 1])
    logits = tf.squeeze(logits, 2, name='logits')
    epsilon = tf.constant(value=0.0000001)
    logits = logits + epsilon
    
    tf.check_numerics(logits,'numerical problem with logits')
    
    y = tf.nn.sigmoid(logits, name='sigmoid_outputs')
    target_float = tf.to_float(y_)
    cross_entropy_all = tf.nn.sigmoid_cross_entropy_with_logits(logits, target_float)
    cross_entropy = tf.reduce_mean(cross_entropy_all)
    
    return y, cross_entropy

def build_net(x, n_inputs, layer_sizes, use_batch_norm, keep_prob):  
    
    previous_layer = x
    
#     for n_units in layer_sizes:  
#         W = weight_variable([n_inputs, n_units])
#         b = bias_variable([n_units])    
#         h = tf.nn.relu(tf.matmul(previous_layer, W) + b)   
#         n_inputs = n_units 
#         previous_layer = h

    n_units = 256
    W_1 = weight_variable([n_inputs, n_units])
    b_1 = bias_variable([n_units])    
    lin_1 = tf.matmul(x, W_1) + b_1   

    if use_batch_norm:
        # Calculate batch mean and variance
        batch_mean1, batch_var1 = tf.nn.moments(lin_1,[0])
    
        # Apply the initial batch normalizing transform
        h_hat = (lin_1 - batch_mean1) / tf.sqrt(batch_var1 + 1e-3)
        
        # Create two new parameters, scale and beta (shift)
        scale1 = tf.Variable(tf.ones([n_units]))
        beta1 = tf.Variable(tf.zeros([n_units]))
        
        # Scale and shift to obtain the final output of the batch normalization
        # this value is fed into the activation function (here a sigmoid)
        BN1 = scale1 * h_hat + beta1
        h_1 = tf.nn.relu(BN1)    
    else:
        h_1 = tf.nn.relu(lin_1)

    h_1_drop = tf.nn.dropout(h_1, keep_prob) 

    n_inputs = n_units
    n_units = 128
    W_2 = weight_variable([n_inputs, n_units])
    b_2 = bias_variable([n_units])    
    lin_2 = tf.matmul(h_1_drop, W_2) + b_2   

    if use_batch_norm:
        # Calculate batch mean and variance
        batch_mean2, batch_var2 = tf.nn.moments(lin_2,[0])
    
        # Apply the initial batch normalizing transform
        h_hat_2 = (lin_2 - batch_mean2) / tf.sqrt(batch_var2 + 1e-3)
        
        # Create two new parameters, scale and beta (shift)
        scale1_2 = tf.Variable(tf.ones([n_units]))
        beta1_2 = tf.Variable(tf.zeros([n_units]))
        
        # Scale and shift to obtain the final output of the batch normalization
        # this value is fed into the activation function (here a sigmoid)
        BN2 = scale1_2 * h_hat_2 + beta1_2
        h_2 = tf.nn.relu(BN2)    
    else:
        h_2 = tf.nn.relu(lin_2)

    h_2_drop = tf.nn.dropout(h_2, keep_prob) 

    W = weight_variable([n_units, NUM_CATEGORIES])
    b = bias_variable([NUM_CATEGORIES])
    y = tf.matmul(h_2_drop, W) + b
        
    return y

    
          
def show_all_variables():
    total_count = 0
    for idx, op in enumerate(tf.trainable_variables()):
        shape = op.get_shape()
        count = np.prod(shape)
        print("[%2d] %s %s = %s" % (idx, op.name, shape, count))
        total_count += int(count)
    print("[Total] variable size: %s" % "{:,}".format(total_count))
 
def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

def add_dimension(images):
    return np.expand_dims(images, -1)

def save_model(session, model_name, root_dir):
    if not os.path.exists(root_dir + '/model/'):
        os.mkdir(root_dir + '/model/')
    saver = tf.train.Saver(write_version=1)
    save_path = saver.save(session, root_dir + '/model/' + model_name +'.ckpt')
#     print("Model saved in file: %s" % save_path)

def import_data():

    LocSigDB_sequences =[] #533 entries
    for line in open("LocSigDB.csv"):
        LocSigDB_sequences.append( line.replace('\n', '').replace('x', '') )

    SPdb_sequences = {}

    fasta_sequences = SeqIO.parse(open('SPdb'+'.fasta','r'),'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        SPdb_sequences[name] = sequence     
        
    proteins = []
    
    raw_sequences = []
    unknowns = set()
    for category in ['blind','cyto','mito','secreted','nucleus']:
        fasta_sequences = SeqIO.parse(open(category+'.fasta','r'),'fasta')
        count=0
        unknown_count = 0
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            proteins.append( Protein(category, name, sequence) )
            raw_sequences.append(len(sequence))
#             for c in sequence:
#                 if c not in AA_CODES:
#                     unknowns.add(c)
                    
            count += 1
            if 'X' in sequence:
                unknown_count += 1
        print(category + ' has ' + str(count) + ' proteins, ' + str(unknown_count) + ' with X')
    
    print('Imported data. Max length=' + str(max(raw_sequences)) + \
          ' Min length=' + str(min(raw_sequences)))
    

    return proteins, list(SPdb_sequences.values()), LocSigDB_sequences

def convert_aa_dict_to_n20_vec(d):
    vec = []
    
    for code in AA_CODES:
        value = d.get(code)
        if value is None:
            vec.append(0)
        else:
            vec.append(value)
    return vec    

def build_input_data(proteins, SPdb_sequences, LocSigDB_sequences):
    X = []
    y = []
    x = []
    SPdb_count = 0
    LocSigDB_count = 0
    
    for p in proteins:
#         del x[:]
        x = []
        
        for seq_flag in [WHOLE_SEQUENCE, N_TERMINAL_50, C_TERMINAL_50]:
            sub_p = p.get_sub_sequence(seq_flag)

            x.extend( convert_aa_dict_to_n20_vec(sub_p.get_amino_acids_percent()) )
            x.append( sub_p.aromaticity() )
            x.append( sub_p.instability_index() )
            x.append( np.mean(sub_p.flexibility()) )
            x.append( sub_p.isoelectric_point() )
            (Helix, Turn, Sheet) = sub_p.secondary_structure_fraction()
            x.extend( (Helix, Turn, Sheet) )
#             x.append( p.get_N_terminus_aa() )

            if seq_flag == C_TERMINAL_50:
                x.append( int(sub_p.has_KDEL()) )
                x.append( int(sub_p.has_KKXX()) )
                x.append( int(sub_p.has_PTS()) )
            elif seq_flag == WHOLE_SEQUENCE:
                x.append( sub_p.molecular_weight() )
                x.append( int(sub_p.sequence_length()) )
                x.append( int(sub_p.has_Chelsky_sequence()) )
                x.append( int(sub_p.has_NLS()) )
                
#                 for SPdb_seq in SPdb_sequences:
#                     x.append( int(SPdb_seq in sub_p.get_sequence() ) )
#                     if x[-1] == 1:
#                         SPdb_count += 1

                for LocSigdb_seq in LocSigDB_sequences:
                    x.append( int(LocSigdb_seq in sub_p.get_sequence() ) )
                    if x[-1] == 1:
                        LocSigDB_count += 1
 
        X.append( x )
        y.append( p.get_category() )
    
    print('num LocSigDB matches=%d' %(LocSigDB_count) )
    print('num SPdb matches=%d' %(SPdb_count) )
    
    return np.asarray(X), np.asarray(y)

def normalize(X):
    
#     ftr_means = np.mean(X, axis=0)
#     ftr_stds = np.std(X, axis=0)
    
    med = np.median(X, axis=0)
    diff = X - med
    ab = np.abs(diff)
    std = np.median(ab, axis=0)
    std[std == 0] = 1
    
    X_norm = (X - med)/std
#     X_norm = (X - ftr_means)/ftr_stds
    
    return X_norm, med, std

def run_models(FLAGS):
    print('Tensorflow version: ', tf.VERSION)
    print('PYTHONPATH: ',sys.path)
    print('model: ', FLAGS.model )
    root_dir = os.getcwd()
    summaries_dir = root_dir + '/Summaries';
    fn = ''

    
############## Hyperparameters ################################################
    max_num_epochs = 10000
    dropout_val = 0.5
    learning_rate_val = float(FLAGS.lr)
    use_batch_norm = FLAGS.bn
    
    decay = learning_rate_val / 2e4
    use_peepholes = False; peep_str='' #only for LSTM
    BATCH_SIZE = 512
    
    inputs_data_filename = 'input_data.pi'
    calc_inputs = True
    if calc_inputs:
        # Import data
        proteins, SPdb_sequences, LocSigDB_sequences = import_data()
        X_all, y_all = build_input_data(proteins, SPdb_sequences, LocSigDB_sequences)  
        pi.dump( (X_all, y_all), open( inputs_data_filename, "wb" ) )
    else:
        (X_all, y_all) = pi.load( open( inputs_data_filename, "rb" ) )

    X_all, _, _ = normalize(X_all)
    X_non_blind = []
    y_non_blind = []
    X_blind = []
    for x_i, y_i in zip(X_all, y_all):
        if y_i == 4: #blind
            X_blind.append(x_i)
        else:
            X_non_blind.append(x_i)
#             one_hot_vec = np.eye(NUM_CATEGORIES)[y_i]
#             y_non_blind.append(one_hot_vec)
            y_non_blind.append(y_i)
            
    X_all = np.asarray(X_non_blind)
    y_all = np.asarray(y_non_blind)
    
                
    #Build different model types
    
#     if ps._cell_type=='L':
#         cell = tf.nn.rnn_cell.LSTMCell(ps._num_units, use_peepholes=use_peepholes)
#     elif ps._cell_type=='G':
#         cell = tf.nn.rnn_cell.GRUCell(ps._num_units)
# 
#     if ps._num_layers>1:
#         cell = tf.nn.rnn_cell.MultiRNNCell([cell] * ps._num_layers)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    n_inputs = len(X_all[0])
    
    x = tf.placeholder(tf.float32, [None, n_inputs], name='x')
    y_ = tf.placeholder(tf.int32, [None], name='y_')
    y = build_net(x, n_inputs, [256,64], use_batch_norm, keep_prob ) 

    cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))

#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#     train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cross_entropy)
    

    # Test trained model
    argm_y = tf.to_int32( tf.argmax(y, 1) )
    correct_prediction = tf.equal(argm_y, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

#     tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('CrossEntropy', cross_entropy)
    path_arr = [FLAGS.model, "drop{:.1f}".format(dropout_val), peep_str, 'bs' + str(BATCH_SIZE),
                "lr{:.2g}".format(learning_rate_val)]
    if use_batch_norm:
        path_arr.append('bn')

#     show_all_variables()

    
            
    with tf.Session() as sess:  
        tf.global_variables_initializer().run()    
                       
        if FLAGS.sm is not None or FLAGS.eval: #Restore saved model   
#             fn='P1_1x128_L_drop0.5__bs64_lr2e-06_nt0_47'
            if FLAGS.sm is not None:
                fn=FLAGS.sm
            else:
                fn= FLAGS.model
#             print("\n".join([n.name for n in tf.get_default_graph().as_graph_def().node]))

#             model_file_name = root_dir + '/model/' + fn + '.ckpt'    
            model_file_name = root_dir + '/final_models/' + fn + '.ckpt'  
            print('loading model from: ' + model_file_name)  
            saver2restore = tf.train.Saver(write_version=1)
            saver2restore.restore(sess, model_file_name)
            
        if not FLAGS.eval: #Train new model
            # Merge all the summaries and write them out to file
            merged = tf.summary.merge_all()
            
            summary_file_name = '/'.join(path_arr)
            dir_name = summaries_dir + '/' + summary_file_name;
            train_writer = tf.summary.FileWriter(dir_name + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(dir_name + '/test')
            
            lrs = LearningRateScheduler(decay)
            
            print('Starting Training.........')
    
        #     cv_data = KFold(n_splits=5)
            cv_data = StratifiedKFold(n_splits=5)
            fold = -1
            for train, test in cv_data.split(X_all, y_all):
                fold += 1
        #         print("%s %s" % (train, test))
                X_train = X_all[train]
                y_train = y_all[train]
                X_test = X_all[test]
                y_test = y_all[test]
                
                tf.global_variables_initializer().run()    
                conv_tester = ConvergenceTester(0.001, lookback_window=5, decreasing=True) #stop if converged to within 0.05%
#                 clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#                 clf.score(X_test, y_test)                           
                clf = svm.SVC(kernel='rbf', C=1, class_weight='balanced')
                scores = cross_val_score(clf, X_all, y_all, cv=2)
                print('SVM score: ', scores)
    
                db = DataBatcher(X_train, y_train)
                ntrain = X_train.shape[0]
            
                # Train
                for epoch in range(max_num_epochs):
                    start = timer()
               
                    for i in range(ntrain // BATCH_SIZE):
                        learning_rate_val = lrs.get_learning_rate(epoch, learning_rate_val)
                        batch_xs, batch_ys = db.next_batch(BATCH_SIZE)
                        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, 
                                                        learning_rate: learning_rate_val, keep_prob: dropout_val})
                    end = timer()
                 
                    if epoch % 5 == 0: #calc intermediate results
         
                        y_vals, train_accuracy, train_loss, train_summary = sess.run([argm_y, accuracy, cross_entropy, merged], feed_dict={x: X_train, y_: y_train, keep_prob: 1.0})                                      
                        argm_y_val, test_accuracy, test_loss, test_summary = sess.run([argm_y, accuracy, cross_entropy, merged], feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})
                        print("f %d, e %d, tr:te accuracy %g : %g loss %g : %g lr %g et %s" % 
                              (fold, epoch, train_accuracy, test_accuracy, train_loss, test_loss, learning_rate_val, str(datetime.timedelta(seconds=end-start))))
                                       
#                         if np.isnan(train_loss) or np.isnan(test_loss):
#                             exit()
                        
                        train_writer.add_summary(train_summary, i)
                        test_writer.add_summary(test_summary, i)
                        
                        if conv_tester.has_converged(test_loss):
                            print('converged after ', epoch, ' epochs')
                            break
                            
                    #save trained model
                    model_file_name = '_'.join(path_arr)+'_'+ str(epoch) #write every epoch
                    save_model(sess, model_file_name, root_dir)
    #             exit()
            
        if FLAGS.eval and True:
            #print final results        
            nsplits = 10
            n = int(X_train.shape[0] / nsplits)
            train_losses = np.zeros((nsplits))
            train_accuracies = np.zeros((nsplits))
            for i in range(nsplits):
                start = i * n
                end = (i+1) * n
                train_losses[i], train_accuracies[i] = sess.run([cross_entropy, accuracy], feed_dict={x: X_train[start:end], y_: y_train[start:end], keep_prob: 1.0})                                      
            test_loss, test_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})                                      
            print("\ntrain loss %.6f train accuracy %.6f" % (np.mean(train_losses), np.mean(train_accuracies)))
            print("\ntest loss %.6f test accuracy %.6f" % (test_loss, test_accuracy))
    
