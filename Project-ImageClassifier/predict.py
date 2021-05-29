# -*- coding: utf-8 -*-
"""
Spyder Editor
----------------------------------------------------
file:   predict.py

class:  udacity - create your own image classifier

author: Daniel Jaensch
email:  daniel.jaensch@gmail.com
data:   2018-09-12
----------------------------------------------------
"""

# ------ imports -----
import optparse
import json
import cnnnetwork as cnn
# --------------------

# -------------------------------------------------------------------
# ----------------------- predict -----------------------------------
# -------------------------------------------------------------------

parser = optparse.OptionParser("predict.py image_directory checkpoint [Options]")
parser.add_option('--top_k', action="store", dest="top_k", default=5, type="int", help="return top K most likely classes")
parser.add_option('--category_names', action="store", dest="category_names", default="cat_to_name.json", type="str", help="choose a file for categories to real names")
parser.add_option('--gpu', action="store_true", dest="gpu", default=True, help="use GPU for training")

# parse all arguments
options, args = parser.parse_args()

# break: in case there are no command line params
if len( args ) < 2:
    parser.error("basic usage: python predict.py image_directory checkpoint")

# set data_directory from the argument line
param_image_file = args[0]                # default: ./flowers/test/10/image_07117.jpg
param_load_file_name = args[1]            # default: checkpoint.pth
param_output_size = 102                   # 102 - original # 10 - test

print("----- running with params -----")
print("image file:         ", param_image_file)
print("load file:          ", param_load_file_name)

if options.top_k is not None:
    print("top k:              ", options.top_k)
    
if options.category_names is not None:
    print("category names:     ", options.category_names)
    
if options.gpu is not None:
    print("gpu:                ", options.gpu)
print("-------------------------------")


# ------ load dictionary -----------
print("load data dictionary ... ", end="")
with open(options.category_names, 'r') as f:
    cat_to_name = json.load(f)
print("done")
# ----------------------------------

# ------------------ load ----------
model = cnn.load_model( param_load_file_name, options.gpu )
# ----------------------------------

# ------------------ prediction ----
print("--- prediction ---")
top_probs, top_labels, top_flowers = model.predict( param_image_file, options.top_k, cat_to_name )

for i in range( len(top_flowers) ):
    # add +1 to index, because cat_to_name starts with index 1 and not with 0
    print(" {} with {:.3f} is {}".format(i+1, top_probs[i], top_flowers[i] ) )
print("------------------")
# ----------------------------------
# show image or plot bar graph, I couldn't found a good solution to display both things at the same time
#cnn.imshow( cnn.process_image(param_image_file) )
cnn.plot_bargraph( top_probs, top_flowers )


