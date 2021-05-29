# -*- coding: utf-8 -*-
"""
Spyder Editor
----------------------------------------------------
file:   cnn_network.py

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
import datetime

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
# --------------------

# -------------------------------------------------------------------
# -------------- class Neural Network -------------------------------
# -------------------------------------------------------------------
class CNNNetwork(nn.Module):
# ------------ init --------------------------------------
    def __init__(self, param_data_directory, param_output_size, architecture, hidden_units, learning_rate, is_gpu):
        ''' initialize the model and all train/test/valid dataloaders '''
        super(CNNNetwork, self).__init__()
        print("cnn neural network ... ")
        print("load image data ... ", end="")
        # define transforms for the training data and testing data
        self.train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        
        self.test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        
        
        # pass transforms in here, then run the next cell to see how the transforms look
        self.train_data = datasets.ImageFolder( param_data_directory + '/train', transform=self.train_transforms )
        self.test_data = datasets.ImageFolder( param_data_directory + '/test', transform=self.test_transforms )
        self.valid_data = datasets.ImageFolder( param_data_directory + '/valid', transform=self.test_transforms )
        
        self.trainloader = torch.utils.data.DataLoader( self.train_data, batch_size=32, shuffle=True )
        self.testloader = torch.utils.data.DataLoader( self.test_data, batch_size=16 )
        self.validloader = torch.utils.data.DataLoader( self.valid_data, batch_size=16 )
        print("done")
        
        self.class_to_idx = self.test_data.class_to_idx
        
        print("create model ... ", end="")
        self.model = models.__dict__[architecture](pretrained=True) 
        
        # get the correct in-features for the classifier and the first layer
        self.in_features = self.get_in_features_from_model_architecture( architecture )
        
        # this is needed for pre-trained networks
        # freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(OrderedDict([
                                  ('do1', nn.Dropout()),
                                  ('fc1', nn.Linear(self.in_features, hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('do2', nn.Dropout()),
                                  ('fc2', nn.Linear(hidden_units, param_output_size)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
        
        self.model.classifier = self.classifier
        
        # train a model with a pre-trained network
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam( self.model.classifier.parameters(), lr=learning_rate )
        print("done")
        
        if is_gpu and torch.cuda.device_count() > 1:
            print("let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self. model)
            
        print("initialized.")



    def get_in_features_from_model_architecture(self, param_architecure):
        ''' return the correct in-features for the classifier and the first layer 
            based on the choosen architecture 
        '''
        in_features = 0
        
        if "vgg" in param_architecure:
            in_features = self.model.classifier[0].in_features
        elif "densenet" in param_architecure:
            in_features = self.model.classifier.in_features
        elif "resnet" in param_architecure:
            in_features = self.model.fc.in_features
        
        return in_features



    def do_deep_learning(self, epochs, print_every, is_gpu):
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
            self.model.cuda()
    
        self.model.train() # ---------- put model in training mode -------------------
        
        for e in range(0, epochs):
            running_loss = 0
            for ii, (images, labels) in enumerate(self.trainloader):
                steps += 1
    
                if is_gpu:
                    images, labels = images.cuda(), labels.cuda()
                
                images, labels = Variable(images), Variable(labels)
    
                self.optimizer.zero_grad()
    
                # forward and backward passes
                outputs = self.model( images )
                loss = self.criterion( outputs, labels )
                loss.backward()
                self.optimizer.step()
    
                running_loss += loss.item()
    
                # ----- output ----
                if steps % print_every == 0:
                    # make sure network is in eval mode for inference
                    self.model.eval() # ------------- put model in evaluation mode ----------------
                    
                    # turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = self.validation( is_gpu )
                        
                    print("epoch: {}/{}.. ".format( e+1, epochs ),
                          "training loss: {:.3f}.. ".format( running_loss / print_every ),
                          "validation loss: {:.3f}.. ".format( test_loss / len(self.validloader) ),
                          "validation accuracy: {:.3f}".format( accuracy / len(self.validloader) ))
                    
                    running_loss = 0
                    
                    # make sure training is back on
                    self.model.train() # ---------- put model in training mode -------------------
                # ----------------- 
        print("-- done --")
    
    
    # implement a function for the validation pass
    def validation(self, is_gpu):
        ''' calculate the validation based on the valid-files and return the test-loss and the accuracy '''
        test_loss = 0
        accuracy = 0
        
        if is_gpu:
            self.model.cuda()
            
        for images, labels in self.validloader:
            if is_gpu:
                images, labels = images.cuda(), labels.cuda()
                    
            output = self.model( images )
            test_loss += self.criterion( output, labels ).item()
    
            ps = torch.exp( output )
            equality = ( labels.data == ps.max(dim=1)[1] ) # give the highest probability
            accuracy += equality.type( torch.FloatTensor ).mean()
        
        return test_loss, accuracy
        
    
    
    def check_accuracy_on_test(self, is_gpu):
        ''' calculate the accuracy based on the test-files and print it out in percent '''
        print("calculate accuracy on test ... ", end="")
        correct = 0
        total = 0
        
        if is_gpu:
            self.model.cuda()
        
        self.model.eval() # ------------- put model in evaluation mode ----------------
        
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                
                if is_gpu:
                    images, labels = images.cuda(), labels.cuda()
                        
                outputs = self.model( images )
                _, predicted = torch.max( outputs.data, 1 )
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print("done.")
        print('accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    

    
    def predict(self, image_file, topk, cat_to_name ):
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
        self.model.eval() # ------------- put model in evaluation mode ----------------
        
        with torch.no_grad():
            image_variable = Variable( img_tensor )
            outputs = self.model( image_variable )
        
        # top probs
        top_probs, top_labs = outputs.topk( topk )
        top_probs = torch.exp( top_probs )
        top_probs = top_probs.detach().numpy().tolist()[0] 
        top_labs = top_labs.detach().numpy().tolist()[0]
        
        # convert indices to classes, flip it around
        idx_to_class = {val: key for key, val in self.class_to_idx.items()}
    
        top_labels = [ idx_to_class[lab] for lab in top_labs ]
        top_flowers = [ cat_to_name[ idx_to_class[lab] ] for lab in top_labs ]
    
        print("done.")
        return top_probs, top_labels, top_flowers
        
    
    
    def load_state_dictionary(self, state_dictionary):
        ''' helper function to load_state_dict '''
        self.model.load_state_dict(state_dictionary)
    
    
    
    def save_model(self, filename, data_directory, architecture, hidden_units, output_size, epochs, learning_rate ):
        ''' save the trained model in a file '''
        print("save model to: ", filename, end="")
        checkpoint = {'arch': architecture,
                      'in_features': self.in_features,
                      'hidden_units': hidden_units,
                      'learning_rate': learning_rate,
                      'output_size': output_size,
                      'data_directory': data_directory,
                      'epochs': epochs,
                      'optimizer_state_dict': self.optimizer.state_dict,
                      'class_to_idx': self.class_to_idx,
                      'state_dict': self.model.state_dict()}
        torch.save(checkpoint, filename)
        print(" ... done")


# -------------------------------------------------------------------
# -------------- helper functions -----------------------------------
# -------------------------------------------------------------------
def get_duration_in_time( duration ):
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
    model = CNNNetwork(checkpoint['data_directory'], checkpoint['output_size'], checkpoint['arch'], checkpoint['hidden_units'], checkpoint['learning_rate'], is_gpu)
    model.load_state_dictionary(checkpoint['state_dict'])
    print(" ... done")
    
    return model


def plot_bargraph( np_probs, np_flower_names ):
    ''' plot an bar graph '''
    ''' plot an bar graph '''
    y_pos = np.arange( len(np_flower_names) )
    
    plt.barh(y_pos, np_probs, align='center', alpha=0.5)
    plt.yticks(y_pos, np_flower_names)
    plt.gca().invert_yaxis()        # invert y-axis to show the highest prob at the top position
    plt.xlabel("probability from 0 to 1.0")
    plt.title("flowers")
    plt.show()


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

# -------------------------------------------------------------------
# -------------------------------------------------------------------