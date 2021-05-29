# -*- coding: utf-8 -*-
"""
Spyder Editor
----------------------------------------------------
file:   train.py

class:  udacity - create your own image classifier

author: Daniel Jaensch
email:  daniel.jaensch@gmail.com
data:   2018-09-12
----------------------------------------------------
"""

# ------ imports -----
import optparse
import time
import cnnnetwork as cnn
# --------------------

# -------------------------------------------------------------------
# ----------------------- train -------------------------------------
# -------------------------------------------------------------------

parser = optparse.OptionParser("train.py data_directory [Options]")
parser.add_option('--save_dir', action="store", dest="save_directory", default="./", type="str", help="set the directory to save checkpoints")
parser.add_option('--arch', action="store", dest="architecture", default="vgg13", type="str", help="choose architecture for example: vgg13")
parser.add_option('--learning_rate', action="store", dest="learning_rate", default=0.001, type="float", help="set hyperparameter: learning rate (0.01)")
parser.add_option('--hidden_units', action="store", dest="hidden_units", default=500, type="int", help="set hyperparameter: hidden units (512)")
parser.add_option('--epochs', action="store", dest="epochs", default=3, type="int", help="set hyperparameter: epochs (20)")
parser.add_option('--gpu', action="store_true", dest="gpu", default=False, help="use GPU for training")

# parse all arguments
options, args = parser.parse_args()

# break: in case there are no command line params
if len( args ) < 1:
    parser.error("basic usage: python train.py data_directory")

# set data_directory from the argument line
param_data_directory = args[0]      # default: flowers
param_output_size = 102             # 102 - original # 10 - test
save_file_name = "checkpoint.pth"

print("----- running with params -----")
print("data directory: ", param_data_directory)

if options.save_directory is not None:
    print("save directory: ", options.save_directory)
    
if options.architecture is not None:
    print("architecture:   ", options.architecture)
    
if options.learning_rate is not None:
    print("learning rate:  ", options.learning_rate)
    
if options.hidden_units is not None:
    print("hidden units:   ", options.hidden_units)
    
if options.epochs is not None:
    print("epochs:         ", options.epochs)

if options.gpu is not None:
    print("gpu:            ", options.gpu)
print("-------------------------------")

# ------- create cnn model -------
cnnmodel = cnn.CNNNetwork(param_data_directory, param_output_size, options.architecture, options.hidden_units, options.learning_rate, options.gpu)

# save time stamp
start_time = time.time()
# --------- training --------
cnnmodel.do_deep_learning( options.epochs, 20, options.gpu )
# ---------------------------
# print duration time
print("duration: ", cnn.get_duration_in_time( time.time() - start_time ) )

# -------------- save -------
cnnmodel.save_model( options.save_directory + save_file_name, param_data_directory, options.architecture, options.hidden_units, param_output_size, options.epochs, options.learning_rate )
# ---------------------------

# ---- test -----------------
# save time stamp
start_time = time.time()
cnnmodel.check_accuracy_on_test( options.gpu )
# print duration time
print("duration: ", cnn.get_duration_in_time( time.time() - start_time ) )
# ---------------------------



