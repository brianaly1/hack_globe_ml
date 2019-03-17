import tensorflow as tf
import network as net
import pickle
import os
import numpy as np
import math

def save_pickles(list, file_name):
    pickle.dump(list, open(file_name, "wb"))

def read_pickle(file_name):
  with open(file_name, 'rb') as f:
    x = pickle.load(f)
  return x

def shuffle(X,Y):
  p = np.random.permutation(len(Y))
  return(X[p],Y[p])

def train(inputs, targets, net, num_iters, mbatch_size, save_dir):
  inputs_ph = tf.placeholder(tf.float32, [None,400])
  targets_ph = tf.placeholder(tf.float32, [None,134])
  errors = [[],[]]
  losses = [[],[]]
  tr_inputs = inputs[0]
  tr_targets = targets[0]
  m = np.shape(tr_inputs)[0]
  mbatch_num = 0
  check_point = 0
  epoch = math.floor(m / mbatch_size) #number of minibatches per epoch
  logits = net.inference(inputs_ph)
  loss_tf = net.loss(logits, targets_ph)
  final_soft = tf.nn.softmax(logits)
  optimizer = tf.train.AdamOptimizer(0.001).minimize(loss_tf)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(num_iters):
      i_batch = int((mbatch_num)*mbatch_size)
      x_mbatch = tr_inputs[i_batch:i_batch + mbatch_size,:] #select a minibatch
      y_mbatch = tr_targets[i_batch:i_batch + mbatch_size,:] 
      sess.run(optimizer, feed_dict={inputs_ph: x_mbatch, targets_ph: y_mbatch})
      #if (mbatch_num == epoch-1): #every epoch
      for j in range (0,2): #training,val
        loss = sess.run(loss_tf,feed_dict={inputs_ph: inputs[j], targets_ph: targets[j]}) 
        prediction = np.argmax(sess.run(final_soft,feed_dict={inputs_ph: inputs[j], targets_ph: targets[j]}),axis = 1)
        error = np.mean(np.argmax(targets[j],axis = 1) != prediction) 
        losses[j].append(loss)
        errors[j].append(error)
        print("Error is: {}".format(error))
        print("Loss is: {}".format(loss))
      if (mbatch_num == epoch-1): #every epoch
        tr_inputs,tr_targets = shuffle(tr_inputs,tr_targets)
        print('----------------------------')
      mbatch_num = mbatch_num + 1
      mbatch_num = mbatch_num % (epoch)
      if((i+1)%100==0):
        check_point = check_point + 1
        saver.save(sess, save_dir, global_step=i)
  return losses,errors

def infer(inputs, check_dir, nn):
  inputs_ph = tf.placeholder(tf.float32, [None,400])
  logits = nn.inference(inputs_ph)
  final_soft = tf.nn.softmax(logits)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess,tf.train.latest_checkpoint(check_dir))
    prediction = np.argmax(sess.run(final_soft,feed_dict={inputs_ph: inputs}),axis = 1)
  return prediction

def train_main():
    input_file = "/home/brianaly/HackTheGlobe/ML/Data/inputs.p"
    input_file_val = "/home/brianaly/HackTheGlobe/ML/Data/inputs_val.p"
    target_file = "/home/brianaly/HackTheGlobe/ML/Data/targets.p"
    target_file_val = "/home/brianaly/HackTheGlobe/ML/Data/targets_val.p"
    inputs = [read_pickle(input_file), read_pickle(input_file_val)]
    targets = [read_pickle(target_file), read_pickle(target_file_val)]
    nn = net.network("fc",[400, 1000, 2000, 1000, 134])
    save_dir = "/home/brianaly/HackTheGlobe/ML/CheckPoints"
    loss_save = os.path.join(save_dir,"losses.p")
    error_save = os.path.join(save_dir,"errors.p")
    num_iters = 100
    mb_size = 64
    losses,errors = train(inputs, targets, nn, num_iters, mb_size, save_dir)
    save_pickles(losses,loss_save)
    save_pickles(errors, error_save)

def infer_main():
    inputs = [np.array([0]*400)]
    nn = net.network("fc",[400, 1000, 2000, 1000, 134])
    output = infer(inputs,"/home/brianaly/HackTheGlobe/ML", nn)
    print(output)



infer_main()