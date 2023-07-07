from scipy import stats
import numpy as np
import cv2
import math
import time
import glob, os
import random
import ctypes
from datetime import datetime
import skimage
import skimage.io
import iio
import fnmatch
from skimage.color import rgb2gray

import torch
import torchvision.transforms as transforms
from collections import OrderedDict
import json

def print_dict(val_losses, suffix="_valLoss", savefile=None):
    val_losses = dict([(k+suffix, v) for k, v in val_losses.items()])
    message = "["
    for k, v in val_losses.items():
        message += '%s: %.3f, ' % (k, v)
    message = message[:-2] + "]"
    print(message)
    if not savefile is None:
        with open(savefile, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

def add_loss(vloss, vname, loss):
    if not vname in vloss.keys():
        vloss[vname] = OrderedDict()
        for k, v in loss.items():
            vloss[vname][k] = [v]
    else:
        for k, v in loss.items():
            vloss[vname][k].append(v)
    return vloss

def save_ordereddict(d,filepath):
    with open(filepath, 'w') as f:
        f.write(json.dumps(list(d.items())))

def load_ordereddict(filepath):
    with open(filepath, 'r') as read_file:
        dvec = json.loads(read_file.read())
    dout = OrderedDict()
    for d in dvec:
        for i in range(int(len(d)/2)):
            dout[d[2*i]] = d[2*i+1]
    return dout

def define_transforms(opt=0, output_size=0):
    """ Defines a transformation (T) and its inverse (iT) to go from
    the dataset to the input of the network.
    iT is used in testing the recurrency or whenever ones needs to go from
    the output of the network back to a numpy image, do some process, and then go back
    to and feed it to the network again.
    """

    transform_list = [transforms.ToTensor()]
    transform_list += [transforms.Lambda(lambda x: 2.*x - 1.)]
    T = transforms.Compose(transform_list)

    transform_list = [transforms.Lambda(lambda x: ((x + 1.)/2.).permute(1, 2, 0).numpy() )]
    iT = transforms.Compose(transform_list)
    return T, iT

def iio_write(arr,path):
    #skimage.io.imsave(path,arr,check_contrast=False)
    iio.write(path, arr)

def iio_read(path):
    #return np.atleast_3d(skimage.io.imread(path))
    return iio.read(path)

def iio_imshow(img):
    img = img.astype(np.uint8)
    c = img.shape[2]
    if c==1:
        r, g, b = img[:, :, 0], img[:, :, 0], img[:, :, 0]
    elif c==4:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    else:
        r,g,b = cv2.split(np.uint8(np.array(img)))
    img_bgr = cv2.merge([b,g,r])
    cv2.imshow('img',img_bgr)
    cv2.waitKey(0)


def get_files_pattern(d, pattern):
    """
    List elements in the directory d with pattern.
    Sort the elements.
    """
    files = os.listdir(d)
    files = fnmatch.filter(files, pattern)
    return sorted(files)

def list_video_files_at_dir(d):
    paths = get_files_pattern(d, '*tiff')
    if len(paths) == 0:
        paths = get_files_pattern(d, '*tif')
    if len(paths) == 0:
        paths = get_files_pattern(d, '*png')
    if len(paths) == 0:
        paths = get_files_pattern(d, '*jpg')
    if len(paths) == 0:
        paths = get_files_pattern(d, '*jpeg')
    if len(paths) == 0:
        paths = get_files_pattern(d, '*raw')
    assert (len(paths) > 0), "%s is empty!" % d
    return [os.path.join(d, p) for p in paths]

def load_image(path, ftype=8):
    """
    Loads an image normalizing its range to [0,1]

    Inputs:
    path  : (string) path to the image
    ftype : (int) number of bits of the image

    Output:
    img : (numpy array) takes values in [0,1]
    """
    return np.asarray(iio_read(path),
                      dtype=np.float32)/(2**float(ftype)-1)


def pathdiff(a,b):
    assert a[:len(b)]==b, "b should be a subfolder/subfile of a"
    res = os.path.dirname(a[len(b):])
    if res[0]=='/':
        return res[1:]
    else:
        return res

def warpedimagefile(wfolder, fromCode, toCode):
    return os.path.join(wfolder,fromCode+'_'+toCode+'.tif')

class CPPbridge(object):
    def __init__(self,libpath):
        self.libBridge = ctypes.cdll.LoadLibrary(libpath)

        self.libBridge.tvl1flow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libBridge.tvl1flow.restype = None

    def TVL1_flow(self, Im1, Im2):
        """ Im1 and Im2 are iio RGB images.
        They will be translated to gray images inside this code.
        """   
        h, w = Im1.shape[:2]
        h1, w1 = Im2.shape[:2]
        assert h1==h and w1==w, "Both images Im1 and Im2 are supposed to share same size"

        I1 = np.zeros(h*w, dtype = ctypes.c_float)
        I2 = np.zeros(h*w, dtype = ctypes.c_float)
        flow = np.zeros(2*h*w, dtype = ctypes.c_float)

        if Im1.shape[2]==3:
            I1[:] = rgb2gray(Im1).flatten()[:]
            I2[:] = rgb2gray(Im2).flatten()[:]
        elif Im1.shape[2] == 4:
            I1[:] = np.mean(Im1, axis=2).flatten()[:]
            I2[:] = np.mean(Im2, axis=2).flatten()[:]
        elif Im1.shape[2] == 1:
            I1[:] = Im1.flatten()[:]
            I2[:] = Im2.flatten()[:]
        
        floatp = ctypes.POINTER(ctypes.c_float)
        self.libBridge.tvl1flow(I1.ctypes.data_as(floatp), I2.ctypes.data_as(floatp), flow.ctypes.data_as(floatp), ctypes.c_int(w), ctypes.c_int(h))
        # flow = np.resize(data, (int(h), int(w), 2))	
        return flow.reshape(2,h,w).transpose(1,2,0)
    
