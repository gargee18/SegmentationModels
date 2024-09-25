# UNet model for Segmentation 
import torch.nn as nn
import torch
# class ConvReLU(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv_op = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(out_ch),
#             nn.Conv2d(out_ch, out_ch, kernel_size = 3, padding = 1),
#             nn.ReLU(),
#             nn.BatchNorm2d(out_ch),
#         )
#     def forward(self,x):
#         return self.conv_op(x)


# class Encoder(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = ConvReLU(in_ch,out_ch)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
#     def forward(self,x):
#         encode= self.conv(x)
#         pool, ind = self.pool(encode)
#         return pool,ind



# class Decoder(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         # self.unpool = nn.Upsample(scale_factor=2)
#         self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
#         self.deconv = ConvReLU(in_ch,out_ch)
    
#     def forward(self, x, ind):
#         unpool = self.unpool(x, ind)
#         decode = self.deconv(unpool)
#         return decode



# class CustomUnet(nn.Module):
#     def __init__(self, in_ch, num_classes):
#         super().__init__()
#         self.encoder_block_1 = Encoder(in_ch, 64)
#         self.encoder_block_2 = Encoder(64, 128)
#         self.encoder_block_3 = Encoder(128, 256)

#         self.bottle_neck = ConvReLU(256,512)

#         self.decoder_block_1 = Decoder(512, 256)
#         self.decoder_block_2 = Decoder(256, 128)
#         self.decoder_block_3 = Decoder(128, 64)

#         self.out = nn.Conv2d(in_channels = 64, out_channels = num_classes, kernel_size = 1)


#     def forward(self, x):
#         # encode
#         e1, ind1 = self.encoder_block_1(x) 
#         e2, ind2 = self.encoder_block_2(e1) 
#         e3, ind3 = self.encoder_block_3(e2)

#         # bottleneck
#         b1 = self.bottle_neck(e3)

#         # decoder
#         d1 = self.decoder_block_1(b1, ind3)
#         d2 = self.decoder_block_2(d1, ind2)
#         d3 = self.decoder_block_3(d2, ind1)

#         # classification layer
#         output = self.out(d3)  
#         return output





# # Using locally saved pretrained model
class CustomUnet(nn.Module):
    def __init__(self):
        super(CustomUnet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            # nn.Conv2d(32, 8, kernel_size=3, padding=1)  # 8 classes
        )
        self.final_layer =nn.Conv2d(32, 8, kernel_size=3, padding=1)  # 8 classes


    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return self.final_layer(x2)


# # ############### This is the test if required #############################
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = CustomUnet().to(device)
# # print(model)

