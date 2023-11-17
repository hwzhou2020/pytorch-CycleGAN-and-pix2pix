"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def gamma(inputs):
    # gamma function
    outputs = torch.exp(torch.lgamma(inputs))
    return outputs

def cal_raw_uncertainty(alpha1,alpha2,beta):
    # variance of asymmetric generalized gaussian distribution
    # alpha1: negative scale parameter
    # alpha2: positive scale parameter
    # beta: shape parameter
    # return: variance

    var1 = alpha1**2 * gamma(3/beta) / gamma(1/beta)
    var2 = alpha2**2 * gamma(3/beta) / gamma(1/beta)
    std = ( ( torch.sqrt(var1) + torch.sqrt(var2) ) / 2 ) 


    return torch.log10(std+1)

def AGGD_bound(alpha1,alpha2,beta):
    # Asymmetric generalized gaussian distribution
    residue = 0
    f_n = beta / (alpha1 + alpha2) / torch.exp(torch.lgamma(1/beta)) * torch.exp( -torch.pow(residue / alpha1, beta) )
    f_p = beta / (alpha1 + alpha2) / torch.exp(torch.lgamma(1/beta)) * torch.exp( -torch.pow(residue / alpha2, beta) )
    f = (f_n * (f_n > f_p) + f_p * (f_n <= f_p)) 
    return torch.max(f)


def cal_ua_range(up_bound,low_bound):
    # calculate the range of uncertainty-aware loss
    # up_bound: upper bound of alpha1, alpha2, and beta
    # low_bound: lower bound of alpha1, alpha2, and beta
    
    # calculate 8 possible combinations of alpha1, alpha2, and beta
    range_ua_list = torch.zeros(2**3)
    aggd_list = torch.zeros(2**3)
    bound_range = [torch.tensor(up_bound),torch.tensor(low_bound)]
    list_choice = ([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1])
    for i,j,k in list_choice:
        range_ua_list[i*2**2 + j*2 + k] = cal_raw_uncertainty(bound_range[i],bound_range[j],bound_range[k]) 
        aggd_list[i*2**2 + j*2 + k] = AGGD_bound(bound_range[i],bound_range[j],bound_range[k])
    
    # find the minimum and maximum of the 8 possible combinations
    ua_range = [torch.max(range_ua_list)-4,torch.min(range_ua_list)]   # clamp by 1e4 #######################################
    aggd_max = torch.max(aggd_list)

    # ua_range_arg = [torch.argmax(range_ua_list),torch.argmin(range_ua_list)]
    # print('ua_range_arg: ',ua_range_arg)
    print('Uncertainty_range: ',ua_range[0].item(),ua_range[1].item())
    # print('ua_range_list: ',range_ua_list)
    print('AGGD_max: ',aggd_max.item())

    return ua_range, aggd_max


def cal_uncertainty(alpha1,alpha2,beta,ua_range):
    
    std = cal_raw_uncertainty(alpha1,alpha2,beta)
    # normalize uncertainty
    std = (std - ua_range[1]) / (ua_range[0] - ua_range[1]) 

    return std.clamp(0,1)