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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--resolution',default=1024,type=int)
    parser.add_argument('--weight_file',default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt",type=str)
    parser.add_argument('--latent_file1',default="latent_W/0.npy")
    parser.add_argument('--latent_file2',default="latent_W/sample.npy")
    parser.add_argument('--latent_file3',default="latent_W/0.npy")
    parser.add_argument('--latent_file4',default="latent_W/sample_01.npy")
    parser.add_argument('--latent_file5',default="latent_W/sample_01.npy")
    parser.add_argument('--latent_file6',default="latent_W/sample_01.npy")
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
    latents_2=np.load(args.latent_file3)
    latents_3=np.load(args.latent_file4)
    latents_4=np.load(args.latent_file5)
    latents_5=np.load(args.latent_file6)



    latents_0=torch.tensor(latents_0).to(device)
    latents_1=torch.tensor(latents_1).to(device)
    latents_2=torch.tensor(latents_2).to(device)
    # latents_3=torch.tensor(latents_3).to(device)
    # latents_4=torch.tensor(latents_4).to(device)
    # latents_5=torch.tensor(latents_5).to(device)

    print('最初に選んだ元画像の比率はどのくらいにしますか？')
    i = int(input())
    print('2番目に選んだ元画像の比率はどのくらいにしますか？')
    j = int(input())
    print('3番目に選んだ元画像の比率はどのくらいにしますか？')
    k = int(input())
    # print('4番目に選んだ元画像の比率はどのくらいにしますか？')
    # l = int(input())
    # print('5番目に選んだ元画像の比率はどのくらいにしますか？')
    # m = int(input())
    # print('6番目に選んだ元画像の比率はどのくらいにしますか？')
    # n = int(input())


    sigma = i + j + k

    i = i/sigma
    j = j/sigma
    k = k/sigma
    # l = l/sigma
    # m = m/sigma
    # n = n/sigma

    #alpha=(1/100)*i
    #latents=alpha*latents_0+(1-alpha)*latents_1
    #latents=(latents_0+latents_1+latents_2+latents_3) /4
    latents =  i*latents_0+j*latents_1+k*latents_2 # + l*latents_3+m*latents_4 +n*latents_5
    synth_img=g_synthesis(latents)
    synth_img = (synth_img + 1.0) / 2.0
    #save_image(synth_img.clamp(0,1),"result_image/your_idea_face.png")
    save_image(synth_img.clamp(0,1),"result_image/comicface_face_test_comic.png")
    #save_image(synth_img.clamp(0,1),"result_image/{}.png".format(i))






if __name__ == "__main__":
    main()
