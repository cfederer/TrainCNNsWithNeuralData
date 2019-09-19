    import os
import numpy as np
import tensorflow as tf
import scipy 
from CORNetZ_V import CORNetZV
from datagenerator import ImageDataGenerator
from datagenerator_v import ImageDataGeneratorV
from datetime import datetime
from label_maps import * 
from tensorflow.data import Iterator
import pandas as pd
import time
import argparse
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def compute_cost_V(act1, act2, Y):
    cost = 0
    for i in range(act1.shape[0]):
        with tf.Session() as sess: 
            rsa = tf.losses.cosine_distance(tf.nn.l2_normalize(act1[i], 0), tf.nn.l2_normalize(act2[i],0), axis=0)
            c = np.sum(abs(rsa - Y[i]))
            cost += c 
    return cost

def get_coarse_accuracy(fine_preds, fine_labels):
    coarse_preds = list()
    coarse_labels = list()
    n = len(fine_preds)
    for i in range(n):
        coarse_preds.append(coarselabel_from_fineidx(fine_preds[i]))
        coarse_labels.append(coarselabel_from_fineidx(fine_labels[i]))

    accuracy = sum(1 for x,y in zip(coarse_preds, coarse_labels) if x==y) / (float(len(fine_preds)))
    return accuracy

def get_total_cost(cost_v, cost_cifar, lam):
    return ((lam * cost_v) + (cost_cifar))

def get_args(which, alpha):
    args = {}
    args['std_initialization'] = False
    args['train_file'] = 'train.txt'
    args['val_file'] = 'val.txt'
    args['v1_file'] = 'V1_train.txt'
    args['v4_file'] = 'V4.txt'
    args['it_file'] = 'IT.txt'
    args['learning_rate'] = 0.01
    args['num_epochs'] = 100
    args['batch_size'] = 128
    args['v_batch_size'] = 50
    args['dropout_rate'] = .5
    args['num_classes'] = 100
    args['alpha_init'] = alpha 
    args['v'] = which 
    args['train_layers'] = ['conv1', 'conv2', 'conv3','conv4', 'conv5','norm1', 'norm2', 'pool1', 'pool2', 'pool5', 'fc8', 'fc7', 'fc6']
    if(which=='V1'):
        args['checkpoint_path'] = "log/CORNetZ_V1/checkpoints"
        args['results_path'] = "log/CORNetZ_V1/results"
        args['v_file'] = args['v1_file']
    elif(which=='V4'):
        args['checkpoint_path'] = "log/CORNetZ_V4/checkpoints"
        args['results_path'] = "log/CORNetZ_V4/results"
        args['v_file'] = args['v4_file']
    elif(which=='IT'):
        args['checkpoint_path'] = "log/CORNetZ_IT/checkpoints"
        args['results_path'] = "log/CORNetZ_IT/results"
        args['v_file'] = args['it_file']
    else:
        args['checkpoint_path'] = "log/CORNetZ/checkpoints"
        args['results_path'] = "log/CORNetZ/results"
    return args 
        
def run_net(args, test): 
    # Path for tf.summary.FileWriter and to store model checkpoints
    print(args) 
    r = round(np.random.randn(), 4)
    
    # Create parent path if it doesn't exist
    if not os.path.isdir(args['checkpoint_path']):
        os.makedirs(args['checkpoint_path'])

    if(args['std_initialization']):
        shuffle = False
        mode = 'inference'

    else:
        shuffle = True
        mode = 'training'

    print('shuffling training data ' + str(shuffle))
        
    # data
    tr_data = ImageDataGenerator(args['train_file'],
                                 mode=mode, 
                                 batch_size=args['batch_size'],
                                 num_classes=args['num_classes'],
                                 shuffle=shuffle)
    val_data = ImageDataGenerator(args['val_file'],
                                  mode='inference',
                                  batch_size=args['batch_size'],
                                  num_classes=args['num_classes'],
                                  shuffle=False)
    if args['v'] != 'None': 
        v_data = ImageDataGeneratorV(args['v_file'],
                                     mode='inference',
                                     batch_size = args['v_batch_size'])

        iterator_v1 = Iterator.from_structure(v_data.data.output_types,
                                              v_data.data.output_shapes)

        next_batch_v1 = iterator_v1.get_next()

        v_init_op = iterator_v1.make_initializer(v_data.data)

        
    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()
    

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)

    # TF placeholder for graph input and output
    if args['v'] != 'None':  
        img1 = tf.placeholder(tf.float32, [args['v_batch_size'], 227, 227, 3])
        img2 = tf.placeholder(tf.float32, [args['v_batch_size'], 227, 227, 3])
        RSA = tf.placeholder(tf.float32, [args['v_batch_size']])
        alpha = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [args['batch_size'], 227, 227, 3])  
    y = tf.placeholder(tf.float32, [args['batch_size'], args['num_classes']])
    keep_prob = tf.placeholder(tf.float32)
    

    # Initialize model
    model = CORNetZV(x, keep_prob, args['num_classes'], args['train_layers'])

    # Link variable to model output
    if(args['v'] == 'V1'):
        pool_tf1 = model.forward_V1(img1)
        pool_tf2 = model.forward_V1(img2)
    elif(args['v'] == 'V4'):
        pool_tf1 = model.forward_V4(img1)
        pool_tf2 = model.forward_V4(img2)
    elif(args['v'] == 'IT'):
        pool_tf1 = model.forward_IT(img1)
        pool_tf2 = model.forward_IT(img2)
        
    score = model.forward()

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        cif_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                    labels=y))

    if args['v'] != 'None': 
        v_cost = compute_cost_V(pool_tf1, pool_tf2, RSA)
        total_cost = get_total_cost(v_cost, cif_cost, alpha)


    # Train op
    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(args['learning_rate'])
    train_op = optimizer.minimize(loss=cif_cost, global_step=tf.train.get_global_step())
    if args['v'] != 'None': 
        train_op_2 = optimizer.minimize(loss=total_cost, global_step=tf.train.get_global_step())

    # Evaluation op: Accuracy of the model
    model_pred = tf.argmax(score, 1)
    act_pred = tf.argmax(y, 1)
    correct_pred = tf.equal(model_pred, act_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(tr_data.data_size/ args['batch_size']))    
    val_batches_per_epoch = int(np.floor(val_data.data_size / args['batch_size']))
    
    if(test):
        print('Running test with 5 batches per epoch')
        train_batches_per_epoch = 2
        val_batches_per_epoch = 2

    #config = tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if(args['v'] != 'None'):
        f_info = 'CORNetZ_' + str(args['v']) + 'alpha=' + str(args['alpha_init']) + '_num_epochs=' + str(args['num_epochs']) + '_' + str(r)
        df = pd.DataFrame(np.zeros((args['num_epochs'], 9)), columns=['train_fine_acc', 'train_coarse_acc', 'test_fine_acc','test_coarse_acc', 'cif_cost_train', 'cif_cost_test', 'v_cost', 'alpha', 'time'])
    else:
       f_info = 'CORNetZ_' + str(args['num_epochs']) + '_' + str(r)
       df = pd.DataFrame(np.zeros((args['num_epochs'], 7)), columns=['train_fine_acc', 'train_coarse_acc', 'test_fine_acc', 'test_coarse_acc', 'cif_cost_train', 'cif_cost_test','time'])
       
    results_f = args['results_path'] + f_info + '.csv'

    
    t0 = time.time()
    if args['v'] != 'None': 
        print('Running ' + str(args['v']) + ' alpha = ' +str(args['alpha_init']))
        alpha_cur = args['alpha_init']
    else:
        print('Running CORNetZ')



    #f.write(str(ratio) + '\n')
    with tf.Session(config=config) as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        if(args['std_initialization']):
            saver.restore(sess, 'log/CORNetZ_initialweights.ckpt')

            print("{} Start training from initial weights...".format(datetime.now()))
        else:
            print("{} Start training...".format(datetime.now()))

        # Loop over number of epochs
        for epoch in range(args['num_epochs']):
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            if args['v'] != 'None':
                alpha_cur = alpha_cur*(1-(epoch/args['num_epochs']))
                print('epoch ' + str(epoch) + ' alpha ' + str(alpha_cur))


            # Initialize iterator with the training dataset
            sess.run(training_init_op)
            if(args['v'] != 'None'):
                sess.run(v_init_op)


            train_fine_acc = 0.
            train_count = 0
            train_coarse_acc = 0
            cif_cost_train = 0
            v_cost_ = 0
            for step in range(train_batches_per_epoch):
                # get next batch of data
                
                img_batch, label_batch = sess.run(next_batch)

                cif_cost_cur = sess.run(cif_cost, feed_dict = {x: img_batch,
                                                                   y: label_batch,
                                                                   keep_prob: 1.})
                cif_cost_train += cif_cost_cur 

                acc, model_preds, act_preds  = sess.run([accuracy, model_pred, act_pred],
                                           feed_dict={x: img_batch,
                                           y: label_batch,
                                           keep_prob: 1.})
                c_acc = get_coarse_accuracy(model_preds, act_preds)
                train_coarse_acc += c_acc
                 
                train_fine_acc += acc
                train_count += 1
            
                if(args['v'] !='None'):
                    # get v1 batches
                    img1_batch, img2_batch, RSAs_batch = sess.run(next_batch_v1)
                    
                    ## calculate costs for lambda vlaue 
                    v_cost_cur = sess.run(v_cost, feed_dict = {img1: img1_batch,
                                                               img2: img2_batch,
                                                               RSA: RSAs_batch})
                    v_cost_ += v_cost_cur
                     
                    ## run v1 training op on total cost
                    sess.run(train_op_2, feed_dict = { img1: img1_batch,
                                                       img2: img2_batch,
                                                       RSA: RSAs_batch,
                                                       alpha: alpha_cur, 
                                                       x: img_batch,
                                                       y: label_batch,
                                                       keep_prob: args['dropout_rate']})

                else: 
                    sess.run(train_op,  feed_dict={x: img_batch,
                                                   y: label_batch,
                                                   keep_prob: args['dropout_rate']})
            train_fine_acc /= train_count
            train_coarse_acc /= train_count
            cif_cost_train /= train_count
            if(v_cost_ != 0): 
                v_cost_ /= train_count 
            # Validate the model on the entire validation set
            print("{} Start validation for epoch= "+str(epoch+1) + ' ' + format(datetime.now()))
            sess.run(validation_init_op)
            test_fine_acc = 0.
            test_count = 0
            test_coarse_acc = 0
            cif_cost_test = 0 
            for _ in range(val_batches_per_epoch):

                img_batch, label_batch = sess.run(next_batch)
                acc, model_preds, act_preds  = sess.run([accuracy, model_pred, act_pred],
                                           feed_dict={x: img_batch,
                                           y: label_batch,
                                           keep_prob: 1.})
                cif_cost_cur = sess.run(cif_cost, feed_dict = {x: img_batch,
                                                                   y: label_batch,
                                                                   keep_prob: 1.})
                cif_cost_test += cif_cost_cur
                c_acc = get_coarse_accuracy(model_preds, act_preds)
                test_coarse_acc += c_acc
                test_fine_acc += acc
                test_count += 1
            test_fine_acc /= test_count
            test_coarse_acc /= test_count
            cif_cost_test /= test_count
            ti = time.time()
            time_run = (ti-t0) / 60
            df['train_fine_acc'].ix[epoch] = train_fine_acc
            df['train_coarse_acc'].ix[epoch] = train_coarse_acc
            df['test_fine_acc'].ix[epoch] = test_fine_acc
            df['test_coarse_acc'].ix[epoch] = test_coarse_acc
            df['cif_cost_train'].ix[epoch] = cif_cost_train
            df['cif_cost_test'].ix[epoch] = cif_cost_test
            if args['v'] != 'None':
                df['v_cost'].ix[epoch] = v_cost_
                df['alpha'].ix[epoch] = alpha_cur 
            df['time'].ix[epoch] = time_run 
            df.to_csv(results_f)

            print("Time to run epoch " + str(epoch+1) + ' : ' + str(round(time_run,2)) + ' minutes')
            
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                           test_fine_acc))
            print("{} Validation Coarse Accuracy = {:.4f}".format(datetime.now(),
                                                       test_coarse_acc))

            print("{} Training Accuracy = {:.4f}".format(datetime.now(),
                                                           train_fine_acc))
            print("{} Training Coarse Accuracy = {:.4f}".format(datetime.now(),
                                                       train_coarse_acc))

            print("{} Validation Cost= {:.4f}".format(datetime.now(),
                                                           cif_cost_test))
            print("{} Training Cost = {:.4f}".format(datetime.now(),
                                                       cif_cost_train))
            # save checkpoint of the model
            checkpoint_name = os.path.join(args['checkpoint_path'],
                                            'weights' + f_info + '_epoch' + str(epoch+1) + '.ckpt')


            #if not args['test']: 
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                        checkpoint_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Settings to run CORNet')
    parser.add_argument('--v', help='visual info', choices=['V1', 'V4', 'IT', 'None'])
    parser.add_argument('--test', help='run quick test')
    parser.add_argument('--n_epochs', help='number of epochs')
    parser.add_argument('--n_times', help='number of times to run')
    parser.add_argument('--step', help='n steps to do v update')
    parser.add_argument('--alphas', help='initial ratio of errors RDM:CE', nargs='*')
    parser.add_argument('--init', help= 'run with standardized initialization')
    
    
    argp = parser.parse_args()

 
    if argp.alphas is not None:
        alphas = argp.alphas
    else:
        alphas = [0] 
    if argp.step is not None:
        step_date = argp.step
    else:
        step_date = 100
    if argp.n_times is not None:
        n_times = argp.n_times
    else:
        n_times = 1 
    if argp.v is not None:
        which = argp.v
    else:
        which = 'None'

    if argp.test is not None:
        test = argp.test
    else:
        test = False
    if argp.n_epochs is not None:
        n_epochs = int(argp.n_epochs)
    else:
        if(test):
            n_epochs = 2
        else:
            n_epochs = 100

    if argp.init is not None:
        std = True
    else:
        std = False 

    for i in range(int(n_times)):
        for alpha in alphas: 
            args = get_args(which=which, alpha = float(alpha))
            args['test'] = test
            args['num_epochs'] = n_epochs
            args['std_initialization'] = std
            run_net(args, test=test)


