import numpy as np
import os
import pandas as pd
from PIL import Image,ImageOps
from tqdm import tqdm
from numpy import asarray
from numpy import savez_compressed
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gc
import random
random.seed(10)
 
import torch.nn as nn
 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, ReLU, Tanh, Sigmoid, BCELoss
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()


img_shape = (256, 256, 3)

dev = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
print(dev)
device = torch.device(dev)
i_array = []
o_array = []
e_array = []
list_name = glob.glob("all_jsons/*.json")
# df = pd.read_csv('all_attributes.csv')
# del df['Unnamed: 0']
# del df['Unnamed: 0.1']
# df = df.replace(np.NaN,-1)
# names = df['ob_file_name']
# del df['ob_file_name']
# mapper = DataFrameMapper([(df.columns, StandardScaler())])
# scaled_features = mapper.fit_transform(df.copy(), 4)
# df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
# df['ob_file_name']= names

# for i in range(len(list_name)):
#     img = Image.open((list_name[i]).replace('all_jsons','gan_input')+'.jpg')
#     img = img.convert('RGB')
#     img = img.resize((256,256))
#     img = np.asarray(img)/255
#     i_array.append(np.transpose(np.float32(img),(2,0,1)))

#     img = Image.open((list_name[i]).replace('all_jsons','gan_output_birth_build')+'.jpg')
#     img = img.convert('RGB')
#     img = img.resize((256,256))
#     img = np.asarray(img)/255
#     o_array.append(np.transpose(np.float32(img),(2,0,1)))

#     file_name = (list_name[i]).replace('all_jsons\\','')

#     trash = df[df['ob_file_name']==file_name]
#     del trash['ob_file_name']
#     encod = trash.values.tolist()
#     # encod = np.asarray(encod)
#     e_array.append(np.asarray(encod))

# savez_compressed('i_arr.npz', i_array)
# savez_compressed('o_arr.npz', o_array)
# savez_compressed('e_arr.npz', e_array)



# plot images in a nxn grid

t = 0
 
def plot_images(real_in, real_out,fake_out):
    
     
    fig = plt.figure(figsize = (5, 3))
    columns = 3
    rows = 5

    # print('real_in',np.shape(real_in))
    # print('real_out',np.shape(real_out))
    # print('fake_out',np.shape(fake_out))
    
    
    for i in range(rows):
        for j in range(columns):
            plt.axis("off")
            fig.add_subplot(rows, columns, 3*i + j + 1)
            if(j==0):
                
                trash1 = np.array((real_in[i]* 255),np.float32)
                print('real_in_i65',trash1)
                plt.imshow(trash1)
            if(j==1):
                trash1 = np.array((real_out[i]* 255),np.float32) 
                plt.imshow(trash1)
            if(j==2):
                trash1 = np.array((fake_out[i]* 255),np.float32)
                plt.imshow(trash1)

    plt.savefig('gan_epoch_plt\\'+str(t)+'.jpg')
    plt.clf()
    



class GansDataset(Dataset):
    
    def _init_(self,e_arr,o_arr,i_arr):
        self.e = e_arr
        self.o = o_arr
        self.i = i_arr
        
    def _len_(self):
        return len(self.e)
    
    def _getitem_(self,idx):
        # img = Image.open((self.list_name[idx]).replace('all_jsons','gan_input')+'.jpg')
        # img = img.convert('RGB')
        # img = img.resize((256,256))
        # img = np.asarray(img)/255
        # input_image = np.transpose(np.float32(img),(2,0,1))

        # img = Image.open((self.list_name[idx]).replace('all_jsons','gan_output_birth_build')+'.jpg')
        # img = img.convert('RGB')
        # img = img.resize((256,256))
        # img = np.asarray(img)/255
        # output_image = np.transpose(np.float32(img),(2,0,1))

        # file_name = (self.list_name[idx]).replace('all_jsons\\','')

        # trash = df[df['ob_file_name']==file_name]
        # del trash['ob_file_name']
        # encod = trash.values.tolist()
        # encod = np.asarray(encod)
        # print('encod',np.shape(encod))

 
 
        return self.i[idx],self.o[idx],self.e[idx]


i_arr = np.load('i_arr.npz')
o_arr = np.load('o_arr.npz')
e_arr = np.load('e_arr.npz')
 
dset = GansDataset(e_arr['arr_0'],o_arr['arr_0'],i_arr['arr_0']) 
batch_size = 8
shuffle = True
 
dataloader = DataLoader(dataset = dset, batch_size = batch_size, shuffle = shuffle)

class encoder_block(Module):
    def _init_(self,layer_in, n_filters, batchnorm=True, drop_last = True):
        super()._init_()
        if batch_size:
            self.encoder = Sequential(
                Conv2d(in_channels = layer_in, out_channels = n_filters, kernel_size = 4, stride = 2, padding = 1, bias=False),
                BatchNorm2d(n_filters),
                LeakyReLU(0.2, inplace=True)
            )
        else:
            self.encoder = Sequential(
                Conv2d(in_channels = layer_in, out_channels = n_filters, kernel_size = 4, stride = 2, padding = 1, bias=False),
                
                LeakyReLU(0.2, inplace=True)
            )
    def forward(self, input):
        return self.encoder(input)




class decoder_block(Module):
    def _init_(self,layer_in, n_filters, dropout=True):
        super()._init_()

        if dropout:

            self.decoder = Sequential(
                ConvTranspose2d(in_channels = layer_in, out_channels = n_filters , kernel_size = 4, stride = 2, padding = 1, bias = False),
            
            BatchNorm2d(num_features = n_filters),

            nn.Dropout(0.5, inplace = True),


            )
        else:
            self.decoder = Sequential(
                ConvTranspose2d(in_channels = layer_in, out_channels = n_filters , kernel_size = 4, stride = 2, padding = 1, bias = False),
            
            BatchNorm2d(num_features = n_filters),

            )
    def forward(self, input):
        return self.decoder(input)
	



class Generator(Module):
    def _init_(self):
        super()._init_()

        # encoder model
        self.e1 = encoder_block(3,64, batchnorm=False)
        self.e2 = encoder_block(64,128)
        self.e3 = encoder_block(128,256)
        self.e4 = encoder_block(256,512)
        self.e5 = encoder_block(512,512)
        self.e6 = encoder_block(512,512)
        self.e7 = encoder_block(512,512)

        self.encoding_linear = nn.Linear(93,16)

        # bottleneck, no batch norm and relu
        self.b = Sequential( 
                Conv2d(in_channels = 516, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias=False),
                ReLU(inplace = True) 
                )
        
        

        # decoder model
        self.d1 = decoder_block(layer_in=512,n_filters=512)
        self.leaky1 = ReLU(inplace = True)
        self.d2 = decoder_block(layer_in=1024,n_filters=512)
        self.leaky2 = ReLU(inplace = True)
        self.d3 = decoder_block(layer_in=1024,n_filters=512)
        self.leaky3 = ReLU(inplace = True)
        self.d4 = decoder_block(layer_in=1024,n_filters=512,dropout=False)
        self.leaky4 = ReLU(inplace = True)
        self.d5 = decoder_block(layer_in=1024,n_filters=256,dropout=False)
        self.leaky5 = ReLU(inplace = True)
        self.d6 = decoder_block(layer_in=512,n_filters=128,dropout=False)
        self.leaky6 = ReLU(inplace = True)
        self.d7 = decoder_block(layer_in=256,n_filters=64,dropout=False)
        self.leaky7 = ReLU(inplace = True)

        # finalconv2dTranspose

        self.final = ConvTranspose2d(in_channels = 128, out_channels = 3 , kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.final_activation = nn.Tanh()

        ## applying weights to all layers
        # self.e1.apply(init_weights)
        # self.e2.apply(init_weights)
        # self.e3.apply(init_weights)
        # self.e4.apply(init_weights)
        # self.e5.apply(init_weights)
        # self.e6.apply(init_weights)
        # self.e7.apply(init_weights)
        # self.d1.apply(init_weights)
        # self.d2.apply(init_weights)
        # self.d3.apply(init_weights)
        # self.d4.apply(init_weights)
        # self.d5.apply(init_weights)
        # self.d6.apply(init_weights)
        # self.d7.apply(init_weights)



    def forward(self,input,encodings):
        e1 = self.e1(input)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        # print('e7',np.shape(e7))
        encodes = self.encoding_linear(encodings)
        encodes_shape = np.shape(encodes)[0]
        encodes = encodes.view(encodes_shape,4,2,2)
        encode_concat = torch.cat((encodes,e7),1)
        b = self.b(encode_concat)

        d1 = self.d1(b)
        concat1 = torch.cat((d1,e7),1)
        relu1 = self.leaky1(concat1)

        d2 = self.d2(relu1)
        concat2 = torch.cat((d2,e6),1)
        relu2 = self.leaky2(concat2)

        d3 = self.d3(relu2)
        concat3 = torch.cat((d3,e5),1)
        relu3 = self.leaky3(concat3)

        d4 = self.d4(relu3)
        concat4 = torch.cat((d4,e4),1)
        relu4 = self.leaky4(concat4)

        d5 = self.d5(relu4)
        concat5 = torch.cat((d5,e3),1)
        relu5 = self.leaky5(concat5)

        d6 = self.d6(relu5)
        concat6 = torch.cat((d6,e2),1)
        relu6 = self.leaky6(concat6)

        d7 = self.d7(relu6)
        concat7 = torch.cat((d7,e1),1)
        relu7 = self.leaky7(concat7)

        second_last = self.final(relu7)
        return self.final_activation(second_last)

# Defining the Discriminator class
 
class Discriminator(Module):
    def _init_(self):
 
        super()._init_()
        self.dis = Sequential(
 
            # input is (1, 256, 256)
            Conv2d(in_channels = 3, out_channels = 32, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32, 128, 128
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = 32, out_channels = 32*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32*2, 8, 8
            BatchNorm2d(32 * 2),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = 32*2, out_channels = 32*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32*4, 4, 4
            BatchNorm2d(32 * 4),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = 32*4, out_channels = 32*8, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 256, 2, 2
            # NOTE: spatial size of this layer is 2x2, hence in the final layer, the kernel size must be 2 instead (or smaller than) 4
            BatchNorm2d(32 * 8),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = 32*8, out_channels = 32*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 256, 2, 2
            # NOTE: spatial size of this layer is 2x2, hence in the final layer, the kernel size must be 2 instead (or smaller than) 4
            BatchNorm2d(32 * 4),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = 32*4, out_channels = 32*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 256, 2, 2
            # NOTE: spatial size of this layer is 2x2, hence in the final layer, the kernel size must be 2 instead (or smaller than) 4
            BatchNorm2d(32 * 2),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels = 32*2, out_channels = 32, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 256, 2, 2
            # NOTE: spatial size of this layer is 2x2, hence in the final layer, the kernel size must be 2 instead (or smaller than) 4
            BatchNorm2d(32),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = 32, out_channels = 1, kernel_size = 2, stride = 2, padding = 0, bias=False),
            # ouput from above layer is b_size, 1, 1, 1
            Sigmoid()
        )
     
    def forward(self, input):
        # print(np.shape(self.dis(input)))
        return self.dis(input)

def init_weights(m):
    if type(m) == ConvTranspose2d:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif type(m) == BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
    elif type(m) == Conv2d:
        nn.init.normal_(m.weight, 0.0, 0.02)

netG = Generator().to(device)
netD = Discriminator().to(device)



netD.apply(init_weights)
netG.apply(init_weights)

opt_D = optim.Adam(netD.parameters(), lr = 0.0002, betas= (0.5, 0.999))
opt_G = optim.Adam(netG.parameters(), lr = 0.0002, betas= (0.5, 0.999))

bcloss = torch.nn.MSELoss()



# TRAINING GANS
epochs = 1000
Total_G_loss = []
Total_D_loss = []
Total_D_real_accuracy = []
Total_D_fake_accuracy = []

for e in tqdm(range(epochs)):
    flip_discriminator = False
    if(random.randint(1,20)==1):
        flip_discriminator = True     
    # pick each batch b of input images: shape of each batch is (32, 3, 256, 256)
    for i, (input_image, output_image,encodings) in tqdm(enumerate(dataloader)):
        gc.collect()
        G_loss_batch = 0
        D_loss_batch = 0
        D_real_total = 0
        D_real_correct = 0
        D_fake_total = 0
        D_fake_correct = 0

        enco_shape = np.shape(encodings)[0]
        encodings = encodings.view(enco_shape,93)
        ##########################
        ## Update Discriminator ##
        ##########################
 
        # Loss on real images
         
        # clear the gradient
       
        opt_D.zero_grad() # set the gradients to 0 at start of each loop because gradients are accumulated on subsequent backward passes
        
        yhat_real = netD(output_image.to(device,dtype=torch.float)).view(-1) # view(-1) reshapes a 4-d tensor of shape (2,1,1,1) to 1-d tensor with 2 values only
        # specify target labels or true labels
        if flip_discriminator:
            target_real = random.uniform(0.0,0.3)*torch.ones(len(output_image))
        else:
            target_real = random.uniform(0.7,1.2)*torch.ones(len(output_image))
        # calculate loss
        # print(yhat_real)
        loss_real = bcloss(yhat_real.cpu(), target_real.cpu())

        for k in range(len(yhat_real)):

            D_real_total+=1
            if(yhat_real[k]>=0.5):
                q1 = 1
            else:
                q1 = 0
            if(target_real[k]>=0.5):
                q3 = 1
            else:
                q3=0
            if(q1==q3):
                D_real_correct+=1

        # calculate gradients -  or rather accumulation of gradients on loss tensor
        loss_real.backward()
    
        # Loss on fake images
        # Step 2: feed noise to G to create a fake img (this will be reused when updating G)
        fake_img = netG(input_image.to(device,dtype=torch.float),encodings.to(device,dtype=torch.float))

        
        # clear gradient
        opt_G.zero_grad()                         
        
            
        yhat_fake = netD.to(device)(fake_img.detach()).view(-1) # .cuda() is essential because our input i.e. fake_img is on gpu but model isnt (runtimeError thrown); detach is imp: Basically, only track steps on your generator optimizer when training the generator, NOT the discriminator. 
        # specify target labels
        if flip_discriminator:
            target_fake = random.uniform(0.7,1.2)*torch.ones(len(input_image))
        else:
            target_fake = random.uniform(0.0,0.3)*torch.ones(len(input_image))
        # calculate loss
        loss_fake = bcloss(yhat_fake.cpu(), target_fake.cpu())
        # calculate gradients
        loss_fake.backward()
        for k in range(len(yhat_fake)):
            D_fake_total+=1
            if(yhat_fake[k]>=0.5):
                q2 = 1
            else:
                q2 = 0
            if(target_fake[k]>=0.5):
                q4 = 1
            else:
                q4 = 0
            if(q2==q4):
                D_fake_correct+=1

        # total error on D
        loss_disc = 0.5*(loss_real + loss_fake)

        D_loss_batch+=loss_disc
    
            

        ##########################
        #### Update Generator ####
        ##########################

            
        # pass fake image through D
        yhat = netD.to(device)(fake_img).view(-1)
        # specify target variables - remember G wants D to think these are real images so label is 1
        if flip_discriminator:
            target = random.uniform(0.0,0.3)*torch.ones(len(fake_img))
        else:
            target = random.uniform(0.7,1.2)*torch.ones(len(fake_img))
        # calculate loss
        loss_gen = bcloss(yhat.cpu(), target.cpu())
        G_loss_batch+=loss_gen
        # calculate gradients
        loss_gen.backward()
        # update weights on G
        opt_G.step()
        # Update weights of D
        opt_D.step()

        break
    Total_D_real_accuracy.append(D_real_correct/D_real_total)
    Total_D_fake_accuracy.append(D_fake_correct/D_fake_total)
    Total_G_loss.append(G_loss_batch.detach().cpu().numpy()/4)
    Total_D_loss.append(D_loss_batch.detach().cpu().numpy()/4)
        
    torch.save({'Generator':netG.state_dict(),'Descriminator':netD.state_dict(),
                'optimizer_G':opt_G.state_dict(),'optimizer_D':opt_D.state_dict(),
                'Discriminator_Loss':Total_D_loss,'Generator_Loss':Total_G_loss,
                'Total_D_real_accuracy':Total_D_real_accuracy,
                'Total_D_fake_accuracy':Total_D_fake_accuracy
               },
               './models.pth')
    plt.plot(list(range(0,e+1)),Total_D_loss,label = 'Desriminator_loss',c = 'C6')
    
    plt.plot(list(range(0,e+1)),Total_G_loss,label = 'Generator_loss',c = 'C8')
    plt.title('LOSS_GRAPH')
    plt.legend()
    
    # plt.show()
    plt.savefig('loss_graph.png')
    plt.clf()
    plt.plot(list(range(0,e+1)),Total_D_real_accuracy,label = 'Desriminator_real_accuracy',c = 'C6')
    
    plt.plot(list(range(0,e+1)),Total_D_fake_accuracy,label = 'Desriminator_fake_accuracy',c = 'C8')
    plt.title('ACCURACY_GRAPH')
    plt.legend()
    
    # plt.show()
    plt.savefig('accuracy_graph.png')
    plt.clf()

        

    
    
        # convert the fake images from (b_size, 3, 255, 255) to (b_size, 255, 255, 3) for plotting 
    fake_img_plot = np.transpose(fake_img.detach().cpu(), (0,2,3,1))
    real_img_plot = np.transpose(output_image.detach().cpu(), (0,2,3,1))
    input_image_plot = np.transpose(input_image.detach().cpu(), (0,2,3,1))
    plot_images(input_image_plot,real_img_plot,fake_img_plot)
    print("********")
    print(" Epoch %d and iteration %d " % (e, i))
    t = t+1