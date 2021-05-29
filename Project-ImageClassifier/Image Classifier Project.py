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
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import time
import datetime
import json

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
# --------------------

# -------------------------------------------------------------------
# -------------------------------------------------------------------

def do_deep_learning(model, trainloader, validloader, optimizer, criterion, epochs, print_every, is_gpu):
    ''' train the model based on the train-files '''
    if is_gpu:
        print("start deep-learning in -gpu- mode ... ")
    else:
        print("start deep-learning in -cpu- mode ... ")

    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda in case it is activated
    if is_gpu:
        model.cuda()

    model.train() # ---------- put model in training mode -------------------

    for e in range(0, epochs):
        running_loss = 0
        for ii, (images, labels) in enumerate( trainloader ):
            steps += 1

            if is_gpu:
                images, labels = images.cuda(), labels.cuda()

            images, labels = Variable(images), Variable(labels)

            optimizer.zero_grad()

            # forward and backward passes
            outputs = model( images )
            loss = criterion( outputs, labels )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # ----- output ----
            if steps % print_every == 0:
                # make sure network is in eval mode for inference
                model.eval() # ------------- put model in evaluation mode ----------------

                # turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation( model, validloader, criterion, is_gpu )

                print("epoch: {}/{}.. ".format( e+1, epochs ),
                      "training loss: {:.3f}.. ".format( running_loss / print_every ),
                      "validation loss: {:.3f}.. ".format( test_loss / len(validloader) ),
                      "validation accuracy: {:.3f}".format( accuracy / len(validloader) ))

                running_loss = 0

                # make sure training is back on
                model.train() # ---------- put model in training mode -------------------
            # -----------------
    print("-- done --")


# implement a function for the validation pass
def validation(model, validloader, criterion, is_gpu):
    ''' calculate the validation based on the valid-files and return the test-loss and the accuracy '''
    test_loss = 0
    accuracy = 0

    # change to cuda in case it is activated
    if is_gpu:
        model.cuda()

    for images, labels in validloader:
        if is_gpu:
            images, labels = images.cuda(), labels.cuda()

        output = model( images )
        test_loss += criterion( output, labels ).item()

        ps = torch.exp( output )
        equality = ( labels.data == ps.max(dim=1)[1] ) # give the highest probability
        accuracy += equality.type( torch.FloatTensor ).mean()

    return test_loss, accuracy



def check_accuracy_on_test(model, testloader, is_gpu):
    ''' calculate the accuracy based on the test-files and print it out in percent '''
    print("calculate accuracy on test ... ", end="")
    correct = 0
    total = 0

    # change to cuda in case it is activated
    if is_gpu:
        model.cuda()

    model.eval() # ------------- put model in evaluation mode ----------------

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if is_gpu:
                images, labels = images.cuda(), labels.cuda()

            outputs = model( images )
            _, predicted = torch.max( outputs.data, 1 )
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("done.")
    print('accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



def predict(model, image_file, topk=5):
    ''' calculate the topk prediction of the given image-file and
    return the probabilities, lables and resolved flower-names
    '''
    # ------ load image data -----------
    img_np = process_image(image_file)
    # ----------------------------------
    print("get prediction ... ", end="")

    # prepare image tensor for prediction
    img_tensor = torch.from_numpy( img_np ).type(torch.FloatTensor)
    # add batch of size 1 to image
    img_tensor.unsqueeze_(0)

    # probs
    model.eval() # ------------- put model in evaluation mode ----------------

    with torch.no_grad():
        image_variable = Variable( img_tensor )
        outputs = model( image_variable )

    # top probs
    top_probs, top_labs = outputs.topk( topk )
    top_probs = torch.exp( top_probs )
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    # convert indices to classes, flip it around
    idx_to_class = {val: key for key, val in class_to_idx.items()}

    top_labels = [ idx_to_class[lab] for lab in top_labs ]
    top_flowers = [ cat_to_name[ idx_to_class[lab] ] for lab in top_labs ]

    print("done.")
    return top_probs, top_labels, top_flowers



def save_model(model, optimizer, filename, data_directory, class_to_idx, architecture, in_features, hidden_units, output_size, epochs, learning_rate):
    ''' save the trained model in a file '''
    print("save model to: ", filename, end="")
    checkpoint = {'arch': architecture,
                  'in_features': in_features,
                  'hidden_units': hidden_units,
                  'learning_rate': learning_rate,
                  'output_size': output_size,
                  'data_directory': data_directory,
                  'epochs': epochs,
                  'optimizer_state_dict': optimizer.state_dict,
                  'class_to_idx': class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filename)
    print(" ... done")



def get_in_features_from_model_architecture(model, param_architecure):
    ''' return the correct in-features for the classifier and the first layer
        based on the choosen architecture
    '''
    in_features = 0

    if "vgg" in param_architecure:
        in_features = model.classifier[0].in_features
    elif "densenet" in param_architecure:
        in_features = model.classifier.in_features
    elif "resnet" in param_architecure:
        in_features = model.fc.in_features

    return in_features


# -------------------------------------------------------------------
# -------------- helper functions -----------------------------------
# -------------------------------------------------------------------
def get_duration_in_time(duration):
    ''' calculate the duration in hh::mm::ss and return it '''
    seconds = int( duration % 60 )
    minutes = int( (duration / 60) % 60 )
    hours   = int( (duration / 3600) % 24 )
    output = "{:0>2}:{:0>2}:{:0>2}".format(hours, minutes, seconds)
    return output



def get_current_date_time():
    ''' return the current date and time '''
    utc_dt = datetime.datetime.now(datetime.timezone.utc) # UTC time
    dt = utc_dt.astimezone() # local time
    return str(dt)



def load_model( filename, is_gpu ):
    ''' load the trained model from the file and create a model from this and return it '''
    print("load model from: ", filename)
    checkpoint = torch.load(filename)

    print("create model ... ", end="")
    model = models.__dict__[checkpoint['arch']](pretrained=True)

    # this is needed for pre-trained networks
    # freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('do1', nn.Dropout()),
                              ('fc1', nn.Linear(checkpoint['in_features'], checkpoint['hidden_units'])),
                              ('relu', nn.ReLU()),
                              ('do2', nn.Dropout()),
                              ('fc2', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'])),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    if is_gpu and torch.cuda.device_count() > 1:
        print("let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print("done")
    print("initialize model ... ", end="")
    model.load_state_dict(checkpoint['state_dict'])
    print("done")

    class_to_idx = checkpoint['class_to_idx']

    return model, class_to_idx


def plot_bargraph( np_probs, np_flower_names ):
    ''' plot an bar graph '''
    y_pos = np.arange( len(np_flower_names) )

    plt.barh(y_pos, np_probs, align='center', alpha=0.5)
    plt.yticks(y_pos, np_flower_names)
    plt.gca().invert_yaxis()        # invert y-axis to show the highest prob at the top position
    plt.xlabel("probability from 0 to 1.0")
    plt.title("flowers")
    plt.show()


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    print("load image data ... ", end="")
    # define transforms for the training data and testing data
    prediction_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    img_pil = Image.open( image )
    img_tensor = prediction_transforms( img_pil )
    print("done")
    return img_tensor.numpy()

# -------------------------------------------------------------------
# ----------------------- train -------------------------------------
# -------------------------------------------------------------------
# ---- set parameters ---------------
param_data_directory = "flowers"            # default: flowers
param_output_size = 102                     # 102 - original # 10 - test
param_save_file_name = "checkpoint.pth"     # checkpoint.pth
param_save_directory = "./"                 # ./
param_architecture = "vgg13"                # densenet121 or vgg13 or resnet18
param_learning_rate = 0.001                 # 0.001
param_hidden_units = 500                    # 500
param_epochs = 3                            # 3
param_gpu = True                            # True or False
# -----------------------------------

print("----- running with params -----")
print("data directory: ", param_data_directory)
print("save directory: ", param_save_directory)
print("architecture:   ", param_architecture)
print("learning rate:  ", param_learning_rate)
print("hidden units:   ", param_hidden_units)
print("epochs:         ", param_epochs)
print("gpu:            ", param_gpu)
print("-------------------------------")

# ------- create cnn model -------
print("cnn neural network ...")
print("  load image data ... ", end="")
# define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder( param_data_directory + '/train', transform=train_transforms )
test_data = datasets.ImageFolder( param_data_directory + '/test', transform=test_transforms )
valid_data = datasets.ImageFolder( param_data_directory + '/valid', transform=test_transforms )

trainloader = torch.utils.data.DataLoader( train_data, batch_size=32, shuffle=True )
testloader = torch.utils.data.DataLoader( test_data, batch_size=16 )
validloader = torch.utils.data.DataLoader( valid_data, batch_size=16 )
print("done")

class_to_idx = test_data.class_to_idx

print("create model ... ", end="")
model = models.__dict__[param_architecture](pretrained=True)

# get the correct in-features for the classifier and the first layer
in_features = get_in_features_from_model_architecture(model, param_architecture )

# this is needed for pre-trained networks
# freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('do1', nn.Dropout()),
                          ('fc1', nn.Linear(in_features, param_hidden_units)),
                          ('relu', nn.ReLU()),
                          ('do2', nn.Dropout()),
                          ('fc2', nn.Linear(param_hidden_units, param_output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

# train a model with a pre-trained network
criterion = nn.NLLLoss()
optimizer = optim.Adam( model.classifier.parameters(), lr=param_learning_rate )
print("done")

if param_gpu and torch.cuda.device_count() > 1:
    print("let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

print("initialized.")
# --------------------------------

# save time stamp
start_time = time.time()
# --------- training --------
do_deep_learning( model, trainloader, validloader, optimizer, criterion, param_epochs, 20, param_gpu )
# ---------------------------
# print duration time
print("duration: ", get_duration_in_time( time.time() - start_time ) )

# -------------- save -------
save_model(model, optimizer, param_save_directory + param_save_file_name, param_data_directory, class_to_idx, param_architecture, in_features, param_hidden_units, param_output_size, param_epochs, param_learning_rate )
# ---------------------------

# ---- test -----------------
# save time stamp
start_time = time.time()
check_accuracy_on_test( model, testloader, param_gpu )
# print duration time
print("duration: ", get_duration_in_time( time.time() - start_time ) )
# ---------------------------

# -------------------------------------------------------------------
# ----------------------- predict -----------------------------------
# -------------------------------------------------------------------
# set data_directory from the argument line
param_image_file = "./flowers/test/10/image_07117.jpg"  # default: flowers/
param_load_file_name = "checkpoint.pth"                 # default: checkpoint.pt
param_top_k = 5                                         # 5
param_category_names = "cat_to_name.json"               # cat_to_name.json
param_gpu = False                                       # True or False

print("----- running with params -----")
print("image file:     ", param_image_file)
print("load file:      ", param_load_file_name)
print("top k:          ", param_top_k)
print("category names: ", param_category_names)
print("gpu:            ", param_gpu)
print("-------------------------------")


# ------ load dictionary -----------
print("load data dictionary ... ", end="")
with open(param_category_names, 'r') as f:
    cat_to_name = json.load(f)
print("done")
# ----------------------------------

# ------------------ load ----------
model, class_to_idx = load_model( param_load_file_name, param_gpu )
# train a model with a pre-trained network
criterion = nn.NLLLoss()
optimizer = optim.Adam( model.classifier.parameters(), lr=param_learning_rate )
# ----------------------------------

# ------------------ prediction ----
print("--- prediction ---")
top_probs, top_labels, top_flowers = predict( model, param_image_file, param_top_k )

for i in range( len(top_flowers) ):
    # add +1 to index, because cat_to_name starts with index 1 and not with 0
    print(" {} with {:.3f} is {}".format(i+1, top_probs[i], top_flowers[i] ) )
print("------------------")
# ----------------------------------
#imshow( process_image(param_image_file) )
plot_bargraph( top_probs, top_flowers )
