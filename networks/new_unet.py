import torch
from torch import nn 
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional
from functools import partial
import opt_einsum as oe
from typing import Optional



@dataclass(eq=False)
class LayerNorm(nn.Module):
    in_channels: int
    eps: float = 1e-6

    def __post_init__(self) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(self.in_channels))
        self.bias = nn.Parameter(torch.zeros(self.in_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)

        return einsum("c, b c ... -> b c ...", self.weight, x) + pad_as(self.bias, x)


@dataclass(eq=False)
class LayerScale(nn.Module):
    in_channels: int
    layerscale_init: Optional[float] = 0.1

    def __post_init__(self) -> None:
        super().__init__()

        if self.layerscale_init is not None: 
            self.layerscale = nn.Parameter(self.layerscale_init * torch.ones(self.in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layerscale_init is not None:
            return einsum("c, b c ... -> b c ...", self.layerscale, x)

        return x

def pad_as(x, ref):
    _, _, *dims = ref.size()

    for _ in range(len(dims)):
        x = x.unsqueeze(dim=-1)

    return x

def zero_pad_features(size, x):
    tmp = torch.zeros(size).to(x.device)

    start_x = int((tmp.size(2) - x.size(2)) / 2)
    start_y = int((tmp.size(3) - x.size(3)) / 2)

    i1, i2 = start_x, start_x + x.size(2)
    j1, j2 = start_y, start_y + x.size(3)
    tmp[:, :, i1:i2, j1:j2] = x

    return tmp

einsum = partial(oe.contract, backend="torch")





@dataclass(eq=False)
class ConvBlock(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int = 7
    padding: int = 3
    layerscale_init: Optional[float] = 0.1

    def __post_init__(self):
        super().__init__()

        if self.in_channels != self.out_channels:
            self.proj = nn.Conv2d(self.in_channels, self.out_channels, 1)
        else:
            self.proj = nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, padding=self.padding, groups=self.out_channels),
            LayerNorm(self.out_channels),
            nn.Conv2d(self.out_channels, 4 * self.out_channels, 1),
            nn.GELU(),
            nn.Conv2d(4 * self.out_channels, self.out_channels, 1)
        )

        self.layerscale = LayerScale(self.out_channels, layerscale_init=self.layerscale_init)
    
    def forward(self, x):
        x = self.proj(x)

        return x + self.layerscale(self.block(x))


@dataclass(eq=False)
class NConvBlock(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int = 7
    padding: int = 3
    layerscale_init: Optional[float] = 0.1
    n_blocks: int = 2

    def __post_init__(self):
        super().__init__()

        self.blocks = nn.Sequential(*[
            ConvBlock(
                self.out_channels if i else self.in_channels, self.out_channels, 
                kernel_size=self.kernel_size,
                padding=self.padding,
                layerscale_init=self.layerscale_init
            )
            for i in range(self.n_blocks)
        ])
    
    def forward(self, x):
        return self.blocks(x)


@dataclass(eq=False)
class UpConv(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int = 7
    padding: int = 3
    layerscale_init: Optional[float] = 0.1
    upsampling_mode: str = "nearest"

    def __post_init__(self):
        super().__init__()

        if self.upsampling_mode in ["nearest", "bilinear", "bicubic"]:
            self.upsampling = nn.Upsample(
                scale_factor=2,
                align_corners=True,
                mode=self.upsampling_mode
            )
        else:
            raise Exception(f'Unknown upsampling mode `{self.upsampling_mode}`.')
    
        self.postconv = ConvBlock(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            layerscale_init=self.layerscale_init
        )
    
    def forward(self, x):
        x = self.upsampling(x)
        x = self.postconv(x)

        return x


@dataclass(eq=False)
class DownConv(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int = 7
    padding: int = 3
    layerscale_init: Optional[float] = 0.1
    downsampling_mode: str = "maxpool"

    def __post_init__(self):
        super().__init__()

        if self.downsampling_mode == "maxpool":
            self.downsampling = nn.MaxPool2d(2)
        
        elif self.downsampling_mode == "avgpool":
            self.downsampling = nn.AvgPool2d(2)
        
        elif self.downsampling_mode == "stridedconv":
            self.downsampling = nn.Conv2d(
                self.in_channels, self.out_channels, 4,
                padding=1, stride=2
            )

        self.postconv = ConvBlock(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            layerscale_init=self.layerscale_init
        )
    
    def forward(self, x):
        x = self.downsampling(x)
        x = self.postconv(x)

        return x


@dataclass(eq=False)
class NewUNet(nn.Module):
    in_channels: int
    out_channels: int
    filters: int = 48
    kernel_size: int = 7
    padding: int = 3
    depth: int = 4
    n_blocks_encoder: int = 2
    n_blocks_decoder: int = 2
    n_blocks_bottleneck: int = 2
    n_blocks_postprocessing: int = 2
    downsampling_mode: str = "maxpool"
    upsampling_mode: str = "bilinear"
    fusion_mode: str = "cat"
    layerscale_init: float = 0.1
    
    def __post_init__(self):
        super().__init__()


        self.f_downsampling = partial(
            DownConv,
            kernel_size=self.kernel_size,
            padding=self.padding,
            downsampling_mode=self.downsampling_mode,
            layerscale_init=self.layerscale_init
        )

        self.f_upsampling = partial(
            UpConv,
            kernel_size=self.kernel_size,
            padding=self.padding,
            upsampling_mode=self.upsampling_mode,
            layerscale_init=self.layerscale_init
        )

        self.f_nconv = partial(
            NConvBlock,
            kernel_size=self.kernel_size,
            padding=self.padding,
            layerscale_init=self.layerscale_init
        )

        self.init_preprocessing()
        self.init_encoder()
        self.init_bottleneck()
        self.init_decoder()
        self.init_postprocessing()

    
    def init_preprocessing(self):
        self.preprocessing = nn.Identity()


    def init_encoder(self):
        self.encoder_convs = nn.ModuleList([
            self.f_nconv(
                self.in_channels, self.filters,
                n_blocks=self.n_blocks_encoder
            )
        ])
        self.encoder_downs = nn.ModuleList([])

        for _ in range(self.depth - 1):
            self.encoder_downs.append(
                self.f_downsampling(self.filters, self.filters)
            )
            self.encoder_convs.append(
                self.f_nconv(
                    self.filters, self.filters,
                    n_blocks=self.n_blocks_encoder
                )
            )


    def init_bottleneck(self):
        self.bottleneck = self.f_nconv(
            self.filters, self.filters,
            n_blocks=self.n_blocks_bottleneck
        )


    def init_decoder(self):
        self.decoder_convs = nn.ModuleList([
            self.f_nconv(
                2 * self.filters if self.fusion_mode == "cat" else self.filters, self.filters,
                n_blocks=self.n_blocks_decoder
            )
            for _ in range(self.depth - 1)
        ])

        self.decoder_ups = nn.ModuleList([
            self.f_upsampling(self.filters, self.filters)
            for _ in range(self.depth - 1)
        ])

        if self.fusion_mode == "sum":
            self.layerscales = nn.ModuleList([
                LayerScale(self.filters, layerscale_init=self.layerscale_init)
                for _ in range(self.depth - 1)
            ])


    def init_postprocessing(self):
        self.postprocessing = nn.Sequential(
            self.f_nconv(
                self.filters, self.filters,
                n_blocks=self.n_blocks_postprocessing
            ),
            nn.Conv2d(self.filters, self.out_channels, 1)
        )


    def fusion(self, x, x_enc, idx):
        if self.fusion_mode == "sum":
            return x + self.layerscales[idx](x_enc)
        
        elif self.fusion_mode == "cat":
            return torch.cat((x, x_enc), 1)
        
        else:
            raise Exception(f'Unknown fusion mode `{self.fusion_mode}`')


    def forward(self, x):
        # Preprocessing
        x = self.preprocessing(x)

        # Encoder
        encoder_memory = []

        for i in range(self.depth):
            x = self.encoder_convs[i](x)
            encoder_memory.append(x)

            if i < self.depth - 1:
                x = self.encoder_downs[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.depth - 1):
            x = self.decoder_ups[i](x)
            x_enc = encoder_memory[-(i + 2)]

            x = zero_pad_features(x_enc.size(), x)

            x = self.fusion(x, x_enc, i)
            x = self.decoder_convs[i](x)

        # Post-processing
        x = self.postprocessing(x)
    
        return x


class NewUNet_feat(NewUNet):
    _name = "feat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.old_features = None
        self.NoPF = 1
    
    def init_preprocessing(self):
        self.preprocessing_layer = self.f_nconv(self.in_channels, self.filters, n_blocks=1)
    
    def preprocessing(self, x):
        if self.old_features is None:
            raise Exception('Old features is None, please call get_rec_nil_features first.')

        y = self.preprocessing_layer(x)
        y = torch.cat([y, self.old_features], 1)

        return y

    def init_encoder(self):
        self.encoder_convs = nn.ModuleList([
            self.f_nconv(
                2 * self.filters, self.filters,
                n_blocks=self.n_blocks_encoder
            )
        ])
        self.encoder_downs = nn.ModuleList([])

        for _ in range(self.depth - 1):
            self.encoder_downs.append(
                self.f_downsampling(self.filters, self.filters)
            )
            self.encoder_convs.append(
                self.f_nconv(
                    self.filters, self.filters,
                    n_blocks=self.n_blocks_encoder
                )
            )

    def init_postprocessing(self):
        self.postprocessing = nn.Sequential(
            self.f_nconv(
                self.filters, self.filters,
                n_blocks=self.n_blocks_postprocessing
            ),
            nn.Conv2d(self.filters, self.out_channels, 1)
        )
        self.postprocessing[-2].register_forward_hook(self.feature_recurrence_hook)

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
