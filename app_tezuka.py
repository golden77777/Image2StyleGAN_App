from flask import Flask, redirect, request,jsonify, render_template
from tensorflow.keras import models
from PIL import Image
from tensorflow.keras.models import load_model
from flask_cors import CORS
from PIL import ImageFile
from tensorflow.keras import backend
#from keras.backend import tensorflow_backend as backend
#import tensorflow.keras
import numpy as np
import sys, os, io
import glob
import tensorflow as tf
import requests

import contextlib
import numpy as np
import matplotlib.pyplot as plt
from stylegan_layers import  G_mapping,G_synthesis
from read_image import image_reader
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision.utils import save_image
from perceptual_model import VGG16_for_Perceptual
import torch.optim as optim
#%matplotlib inline
import matplotlib.image as mpimg
from IPython.display import Image
from PIL import Image

import numpy as np
import cv2
#from image_process import canny
from datetime import datetime
import os
import string
import random


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__, static_folder='templates')
#app = Flask(__name__)
CORS(app)


def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize(imsize)
    img = np.asarray(img)
    img = img / 255.0
    return img


SAVE_DIR = "templates"
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)



def random_str(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])



# http://127.0.0.1:5000/にアクセスしたら、一番最初に読み込まれるページ
@app.route('/',methods=['GET','POST'])
def index_tezuka():
    return render_template('index_tezuka.html', images=os.listdir(SAVE_DIR)[::-1])


@app.route('/templates/<path:path>')
def send_js(path):
    return send_from_directory(SAVE_DIR, path)

@app.route('/make', methods=['POST'])

def test():
        # res1 = request.form['test1']
        res2 = request.form['test2']
        # res3 = request.form['test3']
        # res4 = request.form['test4']
        # res5 = request.form['test5']
        print("test")
        n1 = request.form['tentacles1']
        n2 = request.form['tentacles2']
        # n3 = request.form['tentacles3']
        # n4 = request.form['tentacles4']
        # n5 = request.form['tentacles5']
        #print(res6)

        #print(res1)
        #print(type(res1))
        # a = int(res1)
        b = float(res2)
        # c = int(res3)
        # d = int(res4)
        # e = int(res5)


        # print(a)
        # print(type(a))
        # c = a + b
        # print (c)


        parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
        parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
        parser.add_argument('--resolution',default=1024,type=int)
        parser.add_argument('--weight_file',default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt",type=str)
        parser.add_argument('--latent_file1',default="latent_W/tezuka_pillow_new_true_" + "0" + n1 +".npy")
        parser.add_argument('--latent_file2',default="latent_W/tezuka_pillow_new_true_" + "0" + n2 +".npy")
        parser.add_argument('--latent_file_mean1',default="latent_W/tezuka_pillow_new_true_01.npy")
        parser.add_argument('--latent_file_mean2',default="latent_W/save_pillow_02.npy")
        parser.add_argument('--latent_file_mean3',default="latent_W/save_pillow_03.npy")
        parser.add_argument('--latent_file_mean4',default="latent_W/save_pillow_04.npy")
        parser.add_argument('--latent_file_mean5',default="latent_W/save_pillow_05.npy")
        parser.add_argument('--latent_file_mean6',default="latent_W/save_pillow_06.npy")




        args=parser.parse_args()

        g_all = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        #('truncation', Truncation(avg_latent)),
        ('g_synthesis', G_synthesis(resolution=args.resolution))
        ]))




        g_all.load_state_dict(torch.load(args.weight_file, map_location=device))
        g_all.eval()
        g_all.to(device)


        g_mapping,g_synthesis=g_all[0],g_all[1]


        latents_0=np.load(args.latent_file1)
        latents_1=np.load(args.latent_file2)
        latents_mean_1=np.load(args.latent_file_mean1)
        latents_mean_2=np.load(args.latent_file_mean2)
        latents_mean_3=np.load(args.latent_file_mean3)



        latents_0=torch.tensor(latents_0).to(device)
        latents_1=torch.tensor(latents_1).to(device)
        latents_mean_1=torch.tensor(latents_mean_1).to(device)
        latents_mean_2=torch.tensor(latents_mean_2).to(device)
        latents_mean_3=torch.tensor(latents_mean_3).to(device)

        # print('最初に選んだ元画像の比率はどのくらいにしますか？')
        #i = a
        # print('2番目に選んだ元画像の比率はどのくらいにしますか？')
        j = b
        # print('3番目に選んだ元画像の比率はどのくらいにしますか？')
        #k = c
        # print('4番目に選んだ元画像の比率はどのくらいにしますか？')
        #l = d
        # print('5番目に選んだ元画像の比率はどのくらいにしますか？')
        #m = e
        #sigma = i + j + k + l + m
        mu = (latents_mean_1 + latents_mean_2 + latents_mean_3) / 3

        latents =  latents_0 + j * (latents_1 - mu)
        latents = latents / (1 + j)
        synth_img=g_synthesis(latents)
        synth_img = (synth_img + 1.0) / 2.0
        #save_image(synth_img.clamp(0,1),"result_image/your_idea_face.png")
        save_image(synth_img.clamp(0,1),"templates/new_contents_test_comic12.png")

        ims = []
        #c = str(c)

        img = cv2.imread("templates/new_contents_test_web8.png")
        #img = synth_img.clamp(0,1)
        save_path = os.path.join("new_contents_test_web8" + ".png")
        cv2.imwrite(save_path, img)

        return redirect('/')

        return render_template('index_tezuka.html',name=name)


if __name__ == '__main__':
    app.run(debug=True, port=5003)
