import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=8):
        super(UNet, self).__init__()
        
        # Contracting Path
        self.encoder_conv1 = self.conv_block(input_channels, 64)
        self.encoder_conv2 = self.conv_block(64, 128)
        self.encoder_conv3 = self.conv_block(128, 256)
        self.encoder_conv4 = self.conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Expansive Path
        self.upconv4 = self.upconv(1024, 512)
        self.decoder_conv4 = self.conv_block(1024, 512)
        
        self.upconv3 = self.upconv(512, 256)
        self.decoder_conv3 = self.conv_block(512, 256)

        self.upconv2 = self.upconv(256, 128)
        self.decoder_conv2 = self.conv_block(256, 128)
        
        self.upconv1 = self.upconv(128, 64)
        self.decoder_conv1 = self.conv_block(128, 64)

        # Output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)  # Softmax activation over the channel dimension

    def conv_block(self, in_channels, out_channels):
        """
        Defines a 2D Convolutional block with two Conv layers + ReLU + BatchNorm
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def upconv(self, in_channels, out_channels):
        """
        Defines a 2D upsampling block followed by a convolutional layer.
        """
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Contracting path
        enc1 = self.encoder_conv1(x)
        enc2 = self.encoder_conv2(self.pool(enc1))
        enc3 = self.encoder_conv3(self.pool(enc2))
        enc4 = self.encoder_conv4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Expansive path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)  # Concatenation along channel axis
        dec4 = self.decoder_conv4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder_conv3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder_conv2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder_conv1(dec1)

        # Final layer
        final_output = self.final_conv(dec1)
        return self.softmax(final_output)

# Testing the model
if __name__ == "__main__":
    model = UNet(input_channels=1, num_classes=8)
    x = torch.randn(1, 1, 256, 256)  # Example input (batch_size, channels, height, width)
    output = model(x)
    print(output.shape)  