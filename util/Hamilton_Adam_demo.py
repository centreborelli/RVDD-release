# Pytorch implementation of Hamilton-Adams demosaicing 
#    J. Hamilton Jr. and J. Adams Jr.  Adaptive color plan interpolation 
#    in single sensor color electronic camera, 1997, US Patent 5,629,734.
#
# Copyright (c) 2021 Gabriele Facciolo 
# based on code by  Yu Guo and Qiyu Jin


import torch
from torch import nn 

class HamiltonAdam(nn.Module):
    def __init__(self, pattern):
        super().__init__()

        self.pattern = pattern

        self.mem_mosaic_bayer_mask = {}
        self.mem_algo2_mask = {}

        self.conv_algo1 = nn.Sequential(
            nn.ReplicationPad2d((2, 2, 2, 2)),
            nn.Conv2d(1, 6, kernel_size=5, bias=False)
        )
        self.conv_algo1.requires_grad_(False)
        self.init_algo1()

        self.conv_algo2_chan = nn.Sequential(
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 6, kernel_size=3, bias=False)
        )
        self.conv_algo2_green = nn.Sequential(
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 4, kernel_size=3, bias=False)
        )
        self.conv_algo2_chan.requires_grad_(False)
        self.conv_algo2_green.requires_grad_(False)
        self.init_algo2()

    
    def init_algo1(self):
        weight = torch.zeros(6, 1, 5, 5)
        # Kh
        weight[0, 0, 2, 1] = .5
        weight[0, 0, 2, 3] = .5

        # Kv
        weight[1, 0, 1, 2] = .5
        weight[1, 0, 3, 2] = .5

        # Deltah
        weight[2, 0, 2, 0] = 1.
        weight[2, 0, 2, 2] = -2.
        weight[2, 0, 2, 4] = 1.

        # Deltav
        weight[3, 0, 0, 2] = 1.
        weight[3, 0, 2, 2] = -2.
        weight[3, 0, 4, 2] = 1.

        # Diffh
        weight[4, 0, 2, 1] = 1.
        weight[4, 0, 2, 3] = -1.

        # Diffv
        weight[5, 0, 1, 2] = 1.
        weight[5, 0, 3, 2] = -1.

        self.conv_algo1[1].weight.data = weight
    

    def init_algo2(self):
        weight1 = torch.zeros(6, 1, 3, 3)
        weight2 = torch.zeros(4, 1, 3, 3)
        # Kh
        weight1[0, 0, 1, 0] = .5
        weight1[0, 0, 1, 2] = .5
        
        # Kv
        weight1[1, 0, 0, 1] = .5
        weight1[1, 0, 2, 1] = .5
        
        # Kp
        weight1[2, 0, 0, 0] = .5
        weight1[2, 0, 2, 2] = .5

        # Kn
        weight1[3, 0, 0, 2] = .5
        weight1[3, 0, 2, 0] = .5
        
        # Diffp
        weight1[4, 0, 0, 0] = -1.
        weight1[4, 0, 2, 2] = 1.
        
        # Diffn
        weight1[5, 0, 0, 2] = -1.
        weight1[5, 0, 2, 0] = 1.
        
        # Deltah
        weight2[0, 0, 1, 0] = .25
        weight2[0, 0, 1, 1] = -.5
        weight2[0, 0, 1, 2] = .25

        # Deltav
        weight2[1, 0, 0, 1] = .25
        weight2[1, 0, 1, 1] = -.5
        weight2[1, 0, 2, 1] = .25

        # Deltap
        weight2[2, 0, 0, 0] = 1.
        weight2[2, 0, 1, 1] = -2.
        weight2[2, 0, 2, 2] = 1.

        # Deltan
        weight2[3, 0, 0, 2] = 1.
        weight2[3, 0, 1, 1] = -2.
        weight2[3, 0, 2, 0] = 1.

        self.conv_algo2_chan[1].weight.data = weight1
        self.conv_algo2_green[1].weight.data = weight2
    

    def algo1(self, x, green_mask):
        green_mask = green_mask[None, None]

        # get the raw CFA data
        rawq = x.sum(1, keepdim=True) # [B, 1, 2H, 2W]
        conv_rawq = self.conv_algo1(rawq) # [B, 6, 2H, 2W]

        rawh = conv_rawq[:, 0] - conv_rawq[:, 2] / 4 # [B, 2H, 2W]
        rawv = conv_rawq[:, 1] - conv_rawq[:, 3] / 4 # [B, 2H, 2W]
        CLh = conv_rawq[:, 4].abs() + conv_rawq[:, 2].abs() # [B, 2H, 2W]
        CLv = conv_rawq[:, 5].abs() + conv_rawq[:, 3].abs() # [B, 2H, 2W]

        # this implements the logic assigning rawh  when CLv > CLh
        #                                     rawv  when CLv < CLh;
        #                                     (rawh+rawv)/2 otherwise
        CLlocation = torch.sign(CLh - CLv)
        green = (1 + CLlocation) * rawv / 2 + (1 - CLlocation) * rawh / 2
        green = green[:, None] * (1 - green_mask) + rawq * green_mask # [B, 2H, 2W]

        return green # [B, 1, 2H, 2W]
    

    def algo2(self, green, x_chan, mask_ochan, mode):
        """
        hamilton-adams red channel processing
        """
        # mask
        _, _, H, W = green.size()

        maskGr, maskGb = map(lambda x: x.to(green.device)[None], self.algo2_mask(H, W))
        mask_ochan = mask_ochan[None] # [1, 2*H, 2*W]

        if mode == 2:
            maskGr, maskGb = maskGb, maskGr
        
        conv_mosaic = self.conv_algo2_chan(x_chan)
        conv_green = self.conv_algo2_green(green)

        Ch = maskGr * (conv_mosaic[:, 0] - conv_green[:, 0])
        Cv = maskGb * (conv_mosaic[:, 1] - conv_green[:, 1])
        Cp = mask_ochan * (conv_mosaic[:, 2] - conv_green[:, 2] / 4)
        Cn = mask_ochan * (conv_mosaic[:, 3] - conv_green[:, 3] / 4)
        CLp = mask_ochan * (conv_mosaic[:, 4].abs() + conv_green[:, 2].abs())
        CLn = mask_ochan * (conv_mosaic[:, 5].abs() + conv_green[:, 3].abs())

        CLlocation = torch.sign(CLp - CLn)
        chan = (1 + CLlocation) * Cn / 2 + (1 - CLlocation) * Cp / 2
        chan = (chan + Ch + Cv)[:, None] + x_chan

        return chan


    def algo2_mask(self, H, W):
        code = (H, W)

        if code in self.mem_algo2_mask:
            return self.mem_algo2_mask[code]

        maskGr = torch.zeros(H, W)
        maskGb = torch.zeros(H, W)

        if self.pattern == 'grbg':
            maskGr[0::2, 0::2] = 1
            maskGb[1::2, 1::2] = 1
        elif self.pattern == 'rggb':
            maskGr[0::2, 1::2] = 1
            maskGb[1::2, 0::2] = 1
        elif self.pattern == 'gbrg':
            maskGb[0::2, 0::2] = 1
            maskGr[1::2, 1::2] = 1
        elif self.pattern == 'bggr':
            maskGb[0::2, 1::2] = 1
            maskGr[1::2, 0::2] = 1
        
        self.mem_algo2_mask[code] = (maskGr, maskGb)
        
        return maskGr, maskGb

    def mosaic_bayer_mask(self, H, W):
        code = (H, W)

        if code in self.mem_mosaic_bayer_mask:
            return self.mem_mosaic_bayer_mask[code]
        
        num = [0] * len(self.pattern)
        for idx, x in enumerate(self.pattern):
            if x == 'r':
                num[idx] = 0
            elif x == 'g':
                num[idx] = 1
            elif x == 'b':
                num[idx] = 2
        
        mask = torch.zeros(3, H, W)
        mask[num[0], 0::2, 0::2] = 1
        mask[num[1], 0::2, 1::2] = 1
        mask[num[2], 1::2, 0::2] = 1
        mask[num[3], 1::2, 1::2] = 1

        self.mem_mosaic_bayer_mask[code] = mask

        return mask
    
    def pack_in_one(self, x):
        B, _, H, W = x.size()
        y = torch.zeros(B, 2 * H, 2 * W, dtype=x.dtype, device=x.device)
        y[:, 0::2, 0::2] = x[:, 0]
        y[:, 0::2, 1::2] = x[:, 1]
        y[:, 1::2, 0::2] = x[:, 2]
        y[:, 1::2, 1::2] = x[:, 3]

        return y
    

    def remosaick(self, x):
        B, _, H, W = x.size()
        y = torch.zeros(B, 4, H // 2, W // 2, dtype=x.dtype, device=x.device)

        y[:, 0] = x[:, 1, 0::2, 0::2]
        y[:, 1] = x[:, 2, 0::2, 1::2]
        y[:, 2] = x[:, 0, 1::2, 0::2]
        y[:, 3] = x[:, 1, 1::2, 1::2]

        return y

    
    def forward(self, x):
        """
        Hamilton-Adams demosaicing main function
        mosaic can be a 2D or 3D array with dimensions W,H,C 
        pattern can be: 'grbg', 'rggb', 'gbrg', 'bggr'
        """
        B_orig, _, H, W = x.size()

        x = x.view(-1, 4, H, W)
        B = x.size(0)

        #pack the 4 channels into a single channel CFA
        x_packed = self.pack_in_one(x) # [B, 2H, 2W]

        #put values between 0 and 255 as it was originally for this method
        #vmax = x_packed.view(B, -1).max(dim=1).values
        #vmin = x_packed.view(B, -1).min(dim=1).values
        #vmax, vmin = map(lambda x: x.view(B, 1, 1), (vmax, vmin))

        #x_packed = 255 * (x_packed - vmin) / (vmax - vmin) # [B, 2H, 2W]

        # mosaic and mask (just to generate the mask)
        mask = self.mosaic_bayer_mask(2 * H, 2 * W).to(x.device) # [3, 2H, 2W]

        x_masked = x_packed[:, None] * mask[None] # [B, 3, 2H, 2W]
        # or oe.contract('b h w, c h w -> b c h w', mosaic, mask, backend="torch")

        # green interpolation (implements Algorithm 1)
        green = self.algo1(x_masked, mask[1])

        # Red and Blue demosaicing (implements Algorithm 2)
        red = self.algo2(green, x_masked[:, 0][:, None], mask[2], 1)
        blue = self.algo2(green, x_masked[:, 2][:, None], mask[0], 2)

        # result image
        y = torch.cat((red, green, blue), 1)

        #reput the values in the interval [Min, Max]
        #y = (vmax - vmin)[:, None] * (y / 255) + vmin[:, None]

        return y.view(B_orig, -1, 2 * H, 2 * W)
