# UNet model for Segmentation 
import torch.nn as nn
import torch


testing=True


class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),  
        )
    def forward(self,x):
        return self.conv_op(x)
    
class DeConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),  
        )
    def forward(self,x):
        return self.conv_op(x)
    

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvReLU(in_ch,out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    def forward(self,x):
        encode= self.conv(x)
        pool, ind = self.pool(encode)
        return pool,ind



class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)#scale_factor=2
        self.conv = DeConvReLU(in_ch,out_ch)
    
    def forward(self, x, ind):
        unpool = self.unpool(x, ind)
        convcode = self.conv(unpool)
        return convcode



class CustomUnetWithSkip(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.encoder_block_1 = Encoder(in_ch, 64)
        self.encoder_block_2 = Encoder(64, 128)
        self.encoder_block_3 = Encoder(128, 256)
        self.encoder_block_4 = Encoder(256, 256)

        self.bottle_neck = ConvReLU(256,256)

        self.decoder_block_0 = Decoder(256, 256)
        self.decoder_block_1 = Decoder(256, 128)
        self.decoder_block_2 = Decoder(128, 64)
        self.decoder_block_3 = Decoder(64, 32)

        self.out = nn.Conv2d(in_channels = 32, out_channels = num_classes, kernel_size=3, padding=1 )

    def forward(self, x):
        # encode
        e1, ind1 = self.encoder_block_1(x) 
        e2, ind2 = self.encoder_block_2(e1) 
        e3, ind3 = self.encoder_block_3(e2)
        e4, ind4 = self.encoder_block_4(e3)

        # bottleneck
        b1 = self.bottle_neck(e4)

        # decoder
        d0 = self.decoder_block_0(b1, ind4)
        d1 = self.decoder_block_1(d0, ind3)
        d2 = self.decoder_block_2(d1, ind2)
        d3 = self.decoder_block_3(d2, ind1)

        # classification layer
        output = self.out(d3)  
        return output


from torchsummary import summary
# # ############### This is the test if required #############################
if(testing):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomUnetWithSkip(1,8).to(device)
    print(model)
    summary(model, input_size=(1, 512, 512))
