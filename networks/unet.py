import torch 
from torch import nn
from torch.functional import norm
from torch.nn import functional as F 
from functools import partial 


#########
# Utils #
#########

import inspect, sys

def get_UNet_cls(mode):
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    for _, cls in clsmembers:
        if hasattr(cls, "_name") and getattr(cls, "_name") == mode.lower():
            return cls

    if mode.lower() == "default" or mode.lower() == "concat":
        return UNet
    
    raise Exception(f'Provided mode "{mode}" does not exist.')

class NConvBlock(nn.Module):
    """
    This nn.Module is used as a building block of the U-Net.
    It corresponds to the operation operated at each scale.
    """

    def __init__(
        self, in_channels, out_channels, normalization, activation, n_blocks=2, bias=True,
        mb_conv=False, use_se=False, mb_ratio=2, mb_residual=True, mb_se_ratio=16,
        eco_conv=False, n_eco_conv=1,  eco_conv_skip=False, eco_conv_groups="full"
    ):
        """
        Instantiates the class NConvBlock.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            normalization (nn.Module): The layer to be used for normalization.
            activation (nn.Module): The layer to be used for the activation function
            n_blocks (int): Controls the number of blocks
            bias (bool): Controls if the convolutions should have bias or not.
        """

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        F_Conv = nn.Conv2d 

        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    F_Conv(
                        in_channels if not i else out_channels,
                        out_channels,
                        3, 
                        padding=1, 
                        bias=bias
                    ),
                    normalization(out_channels),
                    activation()
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x):
        x = self.blocks(x)

        return x


class DoubleConvBlock(NConvBlock):
    """
    This nn.Module is used as a building block of the U-Net.
    It correponds to the operation operated at each scale.
    For backward compatibility.
    """
    pass


class UpConv(nn.Module):
    """
    This nn.Module is used as a building block of the U-Net.
    It corresponds to the upsampling operation in a smooth manner to avoid checkerboard artifacts.
    """

    def __init__(
        self, ch_in, ch_out, normalization, activation, upsampling_mode="nearest", bias=True,
        eco_conv=False, n_eco_conv=1, eco_conv_skip=False, eco_conv_groups="full"
    ):
        """
        Instantiates the UpConv class.

        Args:
            ch_in (int): Number of input channels
            ch_out (int): Number of output channels
            normalization (nn.Module): The layer to be used for normalization
            activation (nn.Module): The layer to be used for the activation function
            upsampling_mode (str): The upsampling mode
            bias (bool): Controls if the convolutions should use bias or not.
        """
        super().__init__()
        
        F_Conv = nn.Conv2d 

        if upsampling_mode in ["nearest", "bilinear", "bicubic"]:
            upsampling = partial(
                nn.Upsample, 
                scale_factor=2, 
                mode=upsampling_mode
            )

        elif upsampling_mode[:14].lower() == "transposedconv":
            if len(upsampling_mode) > 14:
                kernel_size = int(upsampling_mode[14:])
            else:
                kernel_size = 2
            padding = int((kernel_size-1)/2)

            upsampling = partial(
                nn.ConvTranspose2d,
                ch_in, ch_in,
                kernel_size, 
                stride=2,
                padding=padding,
                bias=bias
            )
            

        self.up = nn.Sequential(
            upsampling(),
            F_Conv(ch_in, ch_out, 3, padding=1, bias=bias),
		    normalization(ch_out),
			activation()
        )

    def forward(self, x):
        x = self.up(x)

        return x



def zero_pad_features(size, x):
    """
    Spatially pads the input tensor `x` to match the provided size.

    Args:
        size: Desired size for x.
        x: torch.Tensor to be padded.

    Output:
        y: torch.Tensor of size `size`, zero-padded accordingly.
    """

    tmp = torch.zeros(size).to(x.device)
    
    start_x = int((tmp.size(2) - x.size(2)) / 2)
    start_y = int((tmp.size(3) - x.size(3)) / 2)

    tmp[:,:,start_x:x.size(2)+start_x,start_y:x.size(3)+start_y] = x

    return tmp


######################
# Downsampling modes #
######################

class DownsamplingLayer2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding=0, bias=True,
        eco_conv=False, n_eco_conv=1, eco_conv_skip=False, eco_conv_groups="full"
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias 
        self.eco_conv = eco_conv

        self.F_Conv = nn.Conv2d
        

class ConvMaxPool2d(DownsamplingLayer2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = self.F_Conv(
            self.in_channels, 
            self.out_channels, 
            self.kernel_size, 
            padding=self.padding,
            bias=self.bias
        )
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        return self.maxpool(self.conv(x))


class ConvAvgPool2d(DownsamplingLayer2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = self.F_Conv(
            self.in_channels, 
            self.out_channels, 
            self.kernel_size, 
            padding=self.padding,
            bias=self.bias
        )
        self.avgpool = nn.AvgPool2d(2)
    
    def forward(self, x):
        return self.avgpool(self.conv(x))


class WarpMaxPool2d(DownsamplingLayer2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        return self.maxpool(x)

class StridedConv2d(nn.Module):
    def __init__(self, *args, eco_conv=False, **kwargs):
        super().__init__(*args, **kwargs)

        F_Conv = nn.Conv2d

        self.conv = F_Conv(
            self.in_channels, 
            self.out_channels, 
            kernel_size=2, 
            stride=2,
            bias=self.bias
        )
    
    def forward(self, x):
        return self.conv(x)


#################
# UNet #
#################

class UNet(nn.Module):
    """
    UNet network architecture. The fusion of the feature maps coming from the
    encoder and decoders is performed using concatenation. 
    """

    def __init__(
        self, 
        in_channels=3, 
        out_channels=3, 
        filters=48, 
        depth=4, 
        bottleneck_depth=2, 
        post_depth=2,
        downsampling_mode="convmax", 
        upsampling_mode="bilinear",
        activation="relu",
        normalization=None,
        bottleneck_dilation=False,
        n_blocks_encoder=2,
        n_blocks_decoder=2,
        bias=True,
        residual=False,
        eco_conv=False,
        n_eco_conv=1,
        eco_conv_skip=False,
        eco_conv_groups="full",
    ):
        """

        Args:
            in_channels (int): The number of channels of the input.
            out_channels (int): The number of channels of the output.
            filters (int): The number of filters at the beginning. This number will be multiplied by 2 after each downsampling.
            depth (int): The number of downsamplings.
            bottleneck_depth (int): The number of layers of the bottleneck part.
            downsampling_mode (string): The type of downsampling to be used. Default is None, corresponding to nn.MaxPool2d.
            post_depth (int): The number of layers after the UNet.
            normalization (nn.Module): The layer to be used for normalization. Default is None, corresponding to nn.BatchNorm2d.
            bottleneck_dilation (bool): Enables or disables the dilation of the convolutions used in the bottleneck.
            n_blocks_encoder (int): Controls the number of convs per scale in the encoder.
            n_blocks_decoder (int): Controls the number of convs per scale in the decoder.
            bias (bool): Controls if the convolutions should have biases or not.
            residual (bool): Toggles residual learning.
            eco_conv (bool): Replace 3x3 convs by 3x3 followed by 1x1 convs.
        """

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.depth = depth
        self.bottleneck_depth = bottleneck_depth
        self.post_depth = post_depth
        self.downsampling_mode = downsampling_mode
        self.upsampling_mode = upsampling_mode
        self.normalization = normalization
        self.bottleneck_dilation = bottleneck_dilation
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.bias = bias
        self.residual = residual
        self.eco_conv = eco_conv
        self.n_eco_conv = n_eco_conv
        self.eco_conv_skip = eco_conv_skip
        self.eco_conv_groups = eco_conv_groups

        # Default parameters
        if self.downsampling_mode is None:
            self.F_Downsampling = WarpMaxPool2d

        elif self.downsampling_mode == "stridedconv":
            self.F_Downsampling = StridedConv2d

        elif self.downsampling_mode == "convmax":
            self.F_Downsampling = ConvMaxPool2d
        
        elif self.downsampling_mode == "convavg":
            self.F_Downsampling = ConvAvgPool2d

        else:
            raise Exception(f'downsampling_mode `{self.downsampling_mode}` is unknown')
        
        # Prebind the module
        self.F_Downsampling = partial(
            self.F_Downsampling,
            bias=self.bias,
            eco_conv=self.eco_conv
        )

        if self.normalization is None:
            self.normalization = nn.Identity

        if activation == "silu":
            self.activation = partial(nn.SiLU, inplace=True)

        else:
            self.activation = partial(nn.ReLU, inplace=True) 
        
        # Prebind frequently used modules
        self.F_NConvBlock = partial(
            NConvBlock,
            normalization=self.normalization,
            activation=self.activation,
            bias=self.bias,
            eco_conv=self.eco_conv,
            n_eco_conv=self.n_eco_conv,
            eco_conv_skip=self.eco_conv_skip,
            eco_conv_groups=self.eco_conv_groups
        )

        self.F_UpConv = partial(
            UpConv,
            normalization=self.normalization,
            activation=self.activation,
            upsampling_mode=self.upsampling_mode,
            bias=self.bias,
            eco_conv=self.eco_conv,
            n_eco_conv=self.n_eco_conv,
            eco_conv_skip=self.eco_conv_skip,
            eco_conv_groups=self.eco_conv_groups
        )
        
        # Convolution
        self.F_Conv = nn.Conv2d 

        # Preprocessing definition
        self.init_preprocessing()

        # Encoder definition
        self.init_encoder()

        # Bottleneck definition
        self.init_bottleneck()
        
        # Decoder definition 
        self.init_decoder()
        
        # Post-processing definition
        self.init_postprocessing()


    def init_preprocessing(self):
        """
        Initializes the layers using the preprocessing path into `self.preprocessing`
        """

        self.preprocessing = nn.Identity()


    def init_encoder(self):
        """
        Initializes the layers used in the encoder path into `self.EncoderConvs` and `self.EncoderDown`
        """

        self.EncoderConvs = nn.ModuleList([
            self.F_NConvBlock(
                self.in_channels, 
                self.filters,
                n_blocks=self.n_blocks_encoder
            )
        ])
        self.EncoderDown = nn.ModuleList([])

        for i in range(self.depth-1):
            self.EncoderDown.append(
                self.F_Downsampling(
                    self.filters * 2**i, 
                    self.filters * 2**i, 
                    3, 
                    padding=1
                )
            )

            self.EncoderConvs.append(
                self.F_NConvBlock(
                    self.filters * 2**i, 
                    self.filters * 2**(i+1),
                    n_blocks=self.n_blocks_encoder
                )
            )


    def init_bottleneck(self):
        """
        Initializes the layers used in the bottleneck into `self.bottleneck`
        """

        self.bottleneck = nn.ModuleList([])

        for i in range(self.bottleneck_depth):
            if self.bottleneck_dilation:
                self.bottleneck.append(
                    nn.Sequential(
                        self.F_Conv(
                            self.filters * 2**(self.depth-1), 
                            self.filters * 2**(self.depth-1), 
                            kernel_size=3, 
                            padding=2**i, 
                            dilation=2**i,
                            bias=self.bias
                        ),
                        self.activation()
                    )
                )
            else:
                self.bottleneck.append(
                    nn.Sequential(
                        self.F_Conv(
                            self.filters * 2**(self.depth-1), 
                            self.filters * 2**(self.depth-1), 
                            kernel_size=3, 
                            padding=1,
                            bias=self.bias
                        ),
                        self.activation()
                    )
                )


    def init_decoder(self):
        """
        Initializes the layers used in the decoder path into `self.DecoderConvs` and `self.DecoderUp`
        """

        self.DecoderConvs = nn.ModuleList([
            self.F_NConvBlock(
                self.filters * 2**i, 
                self.filters * 2**(i-1),
                n_blocks=self.n_blocks_decoder
            ) 
            for i in reversed(range(1, self.depth))
        ])
        self.DecoderUp = nn.ModuleList([
            self.F_UpConv(
                self.filters * 2**i, 
                self.filters * 2**(i-1)
            ) 
            for i in reversed(range(1, self.depth))
        ])


    def init_postprocessing(self):
        """
        Initializes the layers used in the post-processing path into `self.PostConvs`
        """

        self.PostConvs = nn.ModuleList([
            nn.Sequential(
                self.F_Conv(
                    self.filters, 
                    self.filters, 
                    3, 
                    padding=1, 
                    bias=self.bias
                ),
                self.normalization(self.filters),
                self.activation()
            )
            for _ in range(self.post_depth-1)
        ])
        self.PostConvs.append(
            self.F_Conv(
                self.filters, 
                self.out_channels, 
                1, 
                bias=self.bias
            )
        )


    def fusion(self, x, y, depth_idx):
        """
        Operates the fusion between the feature map coming from the encoder and the one coming from the decoder

        Args:
            - x (torch.Tensor): Feature map coming from the encoder
            - y (torch.Tensor): Feature map coming from the decoder
            - depth_idx (int): Depth index
        """

        return torch.cat((x, y), dim=1)


    def forward(self, x):
        # Preprocessing
        x = self.preprocessing(x)

        # Encoder
        x_s = []
        x_input = x[:,4:,:,:]

        for i in range(self.depth):
            x = self.EncoderConvs[i](x)

            x_s.append(x)
            
            if i < self.depth-1:
                x = self.EncoderDown[i](x)

        # Bottleneck
        d = x_s[-1]

        s = d
        for i in range(self.bottleneck_depth):
            d = self.bottleneck[i](d)
            s = s + d
        d = s

        # Decoder
        for i in range(self.depth-1):
            d = self.DecoderUp[i](d)

            x = x_s[-(i+2)]

            # handling different sizes: zero padding
            d = zero_pad_features(x.size(), d)
            
            d = self.fusion(x, d, i)
            d = self.DecoderConvs[i](d)
        
        # Post-processing
        for layer in self.PostConvs:
            d = layer(d)

        if self.residual:
            return x_input - d

        return d






class UNet_FixedFeatures(UNet):
    """
    UNet where the number of features per depth has been modified to be constant.
    """

    _name = "fixedfeatures"

    def init_encoder(self):
        """
        Initializes the layers used in the encoder path into `self.EncoderConvs` and `self.EncoderDown`
        """

        self.EncoderConvs = nn.ModuleList([
            self.F_NConvBlock(
                self.in_channels, 
                self.filters, 
                n_blocks=self.n_blocks_encoder
            )
        ])
        self.EncoderDown = nn.ModuleList([])

        for _ in range(self.depth-1):
            self.EncoderDown.append(
                self.F_Downsampling(
                    self.filters, 
                    self.filters,
                    3,
                    padding=1
                )
            )

            self.EncoderConvs.append(
                self.F_NConvBlock(
                    self.filters, 
                    self.filters,
                    n_blocks=self.n_blocks_encoder
                )
            )


    def init_bottleneck(self):
        """
        Initializes the layers used in the bottleneck into `self.bottleneck`
        """

        self.bottleneck = nn.ModuleList([])

        for i in range(self.bottleneck_depth):
            if self.bottleneck_dilation:
                self.bottleneck.append(
                    nn.Sequential(
                        self.F_Conv(
                            self.filters, 
                            self.filters, 
                            kernel_size=3, 
                            padding=2**i, 
                            dilation=2**i,
                            bias=self.bias
                        ),
                        self.activation()
                    )
                )
            else:
                self.bottleneck.append(
                    nn.Sequential(
                        self.F_Conv(
                            self.filters, 
                            self.filters, 
                            kernel_size=3, 
                            padding=1,
                            bias=self.bias
                        ),
                        self.activation()
                    )
                )


    def init_decoder(self):
        """
        Initializes the layers used in the decoder path into `self.DecoderConvs` and `self.DecoderUp`
        """

        self.DecoderConvs = nn.ModuleList([
            self.F_NConvBlock(
                self.filters * 2, 
                self.filters,
                n_blocks=self.n_blocks_decoder
            ) 
            for _ in reversed(range(1, self.depth))
        ])
        self.DecoderUp = nn.ModuleList([
            self.F_UpConv(
                self.filters, 
                self.filters
            ) 
            for _ in reversed(range(1, self.depth))
        ])


    def init_postprocessing(self):
        """
        Initializes the layers used in the post-processing path into `self.PostConvs`
        """

        self.PostConvs = nn.ModuleList([
            nn.Sequential(
                self.F_Conv(
                    self.filters, 
                    self.filters, 
                    3, 
                    padding=1, 
                    bias=self.bias
                ),
                self.normalization(self.filters),
                self.activation()
            )
            for _ in range(self.post_depth-1)
        ])
        self.PostConvs.append(
            self.F_Conv(
                self.filters, 
                self.out_channels, 
                1, 
                bias=self.bias
            )
        )




class UNet_FixedFeatures_feat(UNet_FixedFeatures):
    _name = "fixedfeatures+feat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.old_features = None
        self.NoPF = 1

    def init_preprocessing(self):
        self.preprocessing_layer = self.F_Conv(self.in_channels, self.filters, 3, padding=1, bias=self.bias)


    def preprocessing(self, x):
        if self.old_features is None:
            raise Exception('Old features is None, please call get_rec_nil_features first.')

        y = self.preprocessing_layer(x)
        y = torch.cat([y, self.old_features], 1)

        return y


    def init_encoder(self):
        """
        Initializes the layers used in the encoder path into `self.EncoderConvs` and `self.EncoderDown`
        """

        self.EncoderConvs = nn.ModuleList([
            self.F_NConvBlock(
                2*self.filters, 
                self.filters,
                n_blocks=self.n_blocks_encoder
            )
        ])
        self.EncoderDown = nn.ModuleList([])

        for _ in range(self.depth-1):
            self.EncoderDown.append(
                self.F_Downsampling(
                    self.filters, 
                    self.filters,
                    3,
                    padding=1
                )
            )

            self.EncoderConvs.append(
                self.F_NConvBlock(
                    self.filters, 
                    self.filters,
                    n_blocks=self.n_blocks_encoder
                )
            )

    def init_postprocessing(self):
        """
        Initializes the layers used in the post-processing path into `self.PostConvs`
        """

        self.PostConvs = nn.ModuleList([
            nn.Sequential(
                self.F_Conv(
                    self.filters, 
                    self.filters, 
                    3, 
                    padding=1, 
                    bias=self.bias
                ),
                self.normalization(self.filters),
                self.activation()
            )
            for _ in range(self.post_depth-1)
        ])
        self.PostConvs.append(
            self.F_Conv(
                self.filters, 
                self.out_channels, 
                1, 
                bias=self.bias
            )
        )

        self.PostConvs[-2].register_forward_hook(self.feature_recurrence_hook)


    def feature_recurrence_hook(self, layer, input, output):
        self.old_features = output

    def set_rec_features(self, features):
        self.old_features = features[0]

    def get_current_features(self):
        return [self.old_features]

    def get_rec_nil_features(self, B, H, W, device=None, non_blocking=None):
        self.old_features = torch.zeros(B, self.filters, H, W).to(
            device=device, 
            non_blocking=non_blocking
        )
        return [self.old_features]
