import matplotlib.pyplot as plt
import numpy as np
import pickle

def read_pickle(file_name):
  with open(file_name, 'rb') as f:
    x = pickle.load(f)
  return x

def main():
  loss_file = "/home/brianaly/HackTheGlobe/ML/CheckPoints/losses.p"
  error_file = "/home/brianaly/HackTheGlobe/ML/CheckPoints/errors.p"
  loss = read_pickle(loss_file)
  error = read_pickle(error_file)
  fig1 = plt.figure(1)
  train_error, = plt.plot(error[0], label="Training Error")
  val_error, = plt.plot(error[1], label="Validation Error")
  plt.ylabel('Error')
  plt.xlabel('Iteration')
  plt.legend(handles=[train_error, val_error])
  fig1.savefig('/home/brianaly/HackTheGlobe/ML/CheckPoints/Error.png')
  fig2 = plt.figure(2)
  train_loss, = plt.plot(loss[0], label="Training Loss")
  val_loss, = plt.plot(loss[1], label="Validation Loss")
  plt.ylabel('Loss')
  plt.xlabel('Iteration')
  plt.legend(handles=[train_loss, val_loss])
  fig2.savefig('/home/brianaly/HackTheGlobe/ML/CheckPoints/Loss.png')
main()