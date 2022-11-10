import torch
import torch.nn as nn
from models.gan_model import NLayerDiscriminator, NetC


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Encode_Block(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Encode_Block, self).__init__()

        self.conv1 = Res_Block(in_feat, out_feat)
        self.conv2 = Res_Block_identity(out_feat, out_feat)

        self.ca = ChannelAttention(out_feat)
        self.sa = SpatialAttention()

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.ca(outputs) * outputs
        outputs = self.sa(outputs) * outputs
        return outputs


class Decode_Block(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Decode_Block, self).__init__()

        self.conv1 = Res_Block(in_feat, out_feat)
        self.conv2 = Res_Block_identity(out_feat, out_feat)
        self.ca = ChannelAttention(out_feat)
        self.sa = SpatialAttention()

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.ca(outputs) * outputs
        outputs = self.sa(outputs) * outputs
        return outputs


class Conv_Block(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv_Block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.ca = ChannelAttention(out_feat)
        self.sa = SpatialAttention()

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.ca(outputs) * outputs
        outputs = self.sa(outputs) * outputs
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.de_conv = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.de_conv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Res_Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Res_Block, self).__init__()
        self.conv_input = conv1x1(inplanes, planes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.bn3 = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.conv3 = conv1x1(planes, planes)
        self.stride = stride

    def forward(self, x):
        residual = self.conv_input(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Res_Block_identity(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Res_Block_identity, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.bn3 = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.conv3 = conv1x1(planes, planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class AdditionalInput(nn.Module):
    def __init__(self, poolsize, in_ch, out_ch):
        super(AdditionalInput, self).__init__()
        self.AddiInput = nn.Sequential(
            nn.AvgPool2d(poolsize), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )

    def forward(self, x):
        x = self.AddiInput(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        return x


class Share_Encoder(nn.Module):
    def __init__(self, num_channels=1):
        super(Share_Encoder, self).__init__()
        flt = 64
        self.down1 = Encode_Block(num_channels, flt)
        self.down2 = Encode_Block(flt + 32, flt * 2)
        self.down3 = Encode_Block(flt * 2 + 64, flt * 4)
        self.down4 = Encode_Block(flt * 4 + 128, flt * 8)
        self.down_pool1 = nn.Conv2d(flt, flt, kernel_size=3, stride=2, padding=1)
        self.down_pool2 = nn.Conv2d(
            flt * 2, flt * 2, kernel_size=3, stride=2, padding=1
        )
        self.down_pool3 = nn.Conv2d(
            flt * 4, flt * 4, kernel_size=3, stride=2, padding=1
        )
        self.down_pool4 = nn.Conv2d(
            flt * 8, flt * 8, kernel_size=3, stride=2, padding=1
        )
        self.bottom = Encode_Block(flt * 8, flt * 16)
        self.AdditionalInput1 = AdditionalInput(2, 1, 32)
        self.AdditionalInput2 = AdditionalInput(4, 1, 64)
        self.AdditionalInput3 = AdditionalInput(8, 1, 128)

    def forward(self, inputs):
        input2 = self.AdditionalInput1(inputs)  # 128x128
        input3 = self.AdditionalInput2(inputs)  # 64x64
        input4 = self.AdditionalInput3(inputs)  # 32x32
        down1_feat = self.down1(inputs)
        pool1_feat = self.down_pool1(down1_feat)
        pool1_feat = torch.cat([input2, pool1_feat], dim=1)
        down2_feat = self.down2(pool1_feat)
        pool2_feat = self.down_pool2(down2_feat)
        pool2_feat = torch.cat([input3, pool2_feat], dim=1)
        down3_feat = self.down3(pool2_feat)
        pool3_feat = self.down_pool3(down3_feat)
        pool3_feat = torch.cat([input4, pool3_feat], dim=1)
        down4_feat = self.down4(pool3_feat)
        pool4_feat = self.down_pool4(down4_feat)
        bottom_feat = self.bottom(pool4_feat)

        return bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat


class Decoder(nn.Module):
    def __init__(self, num_channels=1):
        super(Decoder, self).__init__()
        flt = 64
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = Decode_Block(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = Decode_Block(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = Decode_Block(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = Decode_Block(flt * 2, flt)

        self.final = nn.Conv2d(flt, num_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.out6 = outconv(512, 1)
        self.out7 = outconv(256, 1)
        self.out8 = outconv(128, 1)
        self.out9 = outconv(64, 1)

    def forward(self, inputs, down4_feat, down3_feat, down2_feat, down1_feat):
        up1_feat = self.up_cat1(inputs, down4_feat)
        up1_feat = self.up_conv1(up1_feat)
        side6 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)(
            up1_feat
        )

        up2_feat = self.up_cat2(up1_feat, down3_feat)
        up2_feat = self.up_conv2(up2_feat)
        side7 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)(
            up2_feat
        )

        up3_feat = self.up_cat3(up2_feat, down2_feat)
        up3_feat = self.up_conv3(up3_feat)
        side8 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)(
            up3_feat
        )

        up4_feat = self.up_cat4(up3_feat, down1_feat)
        up4_feat = self.up_conv4(up4_feat)

        out6 = self.out6(side6)
        out7 = self.out7(side7)
        out8 = self.out8(side8)
        out9 = self.out9(up4_feat)

        my_list = [out6, out7, out8, out9]
        out10 = torch.mean(torch.stack(my_list), dim=0)
        # out10 = self.sigmoid(out10)
        # out10 = self.final(up4_feat)

        # return outputs, up1_feat, up2_feat, up3_feat, up4_feat
        return out6, out7, out8, out9, out10


class Decoder_fix(nn.Module):
    def __init__(self, num_channels=1):
        super(Decoder_fix, self).__init__()
        flt = 64
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = Decode_Block(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = Decode_Block(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = Decode_Block(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = Decode_Block(flt * 2, flt)
        # self.final = nn.Sequential(
        #     nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
        # )
        self.final = nn.Conv2d(flt, num_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.out6 = outconv(512, 1)
        self.out7 = outconv(256, 1)
        self.out8 = outconv(128, 1)
        self.out9 = outconv(64, 1)

    def forward(self, inputs, down4_feat, down3_feat, down2_feat, down1_feat):
        up1_feat = self.up_cat1(inputs, down4_feat)
        up1_feat = self.up_conv1(up1_feat)
        side6 = UpsampleDeterministic(upscale=8)(up1_feat)

        up2_feat = self.up_cat2(up1_feat, down3_feat)
        up2_feat = self.up_conv2(up2_feat)
        side7 = UpsampleDeterministic(upscale=4)(up2_feat)

        up3_feat = self.up_cat3(up2_feat, down2_feat)
        up3_feat = self.up_conv3(up3_feat)
        side8 = UpsampleDeterministic(upscale=2)(up3_feat)

        up4_feat = self.up_cat4(up3_feat, down1_feat)
        up4_feat = self.up_conv4(up4_feat)

        out6 = self.out6(side6)
        out7 = self.out7(side7)
        out8 = self.out8(side8)
        out9 = self.out9(up4_feat)

        my_list = [out6, out7, out8, out9]
        out10 = torch.mean(torch.stack(my_list), dim=0)
        # out10 = self.sigmoid(out10)
        # out10 = self.final(up4_feat)

        # return outputs, up1_feat, up2_feat, up3_feat, up4_feat
        return out6, out7, out8, out9, out10


def upsample_deterministic(x, upscale):
    """
    x: 4-dim tensor. shape is (batch,channel,h,w)
    output: 4-dim tensor. shape is (batch,channel,self. upscale*h,self. upscale*w)
    """
    return (
        x[:, :, :, None, :, None]
        .expand(-1, -1, -1, upscale, -1, upscale)
        .reshape(x.size(0), x.size(1), x.size(2) * upscale, x.size(3) * upscale)
    )


class UpsampleDeterministic(nn.Module):
    def __init__(self, upscale=2):
        """
        Upsampling in pytorch is not deterministic (at least for the version of 1.0.1)
        see https://github.com/pytorch/pytorch/issues/12207
        """
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        """
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,upscale*h,upscale*w)
        """
        return upsample_deterministic(x, self.upscale)


# class Decoder_fix(nn.Module):
#     def __init__(self, num_channels=1):
#         super(Decoder_fix, self).__init__()
#         flt = 64
#         self.up_cat1 = UpConcat(flt * 16, flt * 8)
#         self.up_conv1 = Decode_Block(flt * 16, flt * 8)
#         self.up_cat2 = UpConcat(flt * 8, flt * 4)
#         self.up_conv2 = Decode_Block(flt * 8, flt * 4)
#         self.up_cat3 = UpConcat(flt * 4, flt * 2)
#         self.up_conv3 = Decode_Block(flt * 4, flt * 2)
#         self.up_cat4 = UpConcat(flt * 2, flt)
#         self.up_conv4 = Decode_Block(flt * 2, flt)
#         # self.final = nn.Sequential(
#         #     nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
#         # )
#         self.final = nn.Conv2d(flt, num_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#         self.out6 = outconv(512, 1)
#         self.out7 = outconv(256, 1)
#         self.out8 = outconv(128, 1)
#         self.out9 = outconv(64, 1)
#
#     def forward(self, inputs, down4_feat, down3_feat, down2_feat, down1_feat):
#         up1_feat = self.up_cat1(inputs, down4_feat)
#         up1_feat = self.up_conv1(up1_feat)
#         side6 = UpsampleDeterministic(upscale=8)(up1_feat)
#
#         up2_feat = self.up_cat2(up1_feat, down3_feat)
#         up2_feat = self.up_conv2(up2_feat)
#         side7 = UpsampleDeterministic(upscale=4)(up2_feat)
#
#         up3_feat = self.up_cat3(up2_feat, down2_feat)
#         up3_feat = self.up_conv3(up3_feat)
#         side8 = UpsampleDeterministic(upscale=2)(up3_feat)
#
#         up4_feat = self.up_cat4(up3_feat, down1_feat)
#         up4_feat = self.up_conv4(up4_feat)
#
#         out6 = self.out6(side6)
#         out7 = self.out7(side7)
#         out8 = self.out8(side8)
#         out9 = self.out9(up4_feat)
#
#         my_list = [out6, out7, out8, out9]
#         out10 = torch.mean(torch.stack(my_list), dim=0)
#         # out10 = self.sigmoid(out10)
#         # out10 = self.final(up4_feat)
#
#         # return outputs, up1_feat, up2_feat, up3_feat, up4_feat
#         return out6, out7, out8, out9, out10


class Shared_Res_unet_cbam_insn(nn.Module):
    def __init__(self, mode):
        super(Shared_Res_unet_cbam_insn, self).__init__()
        self.mode = mode
        self.encoder = Share_Encoder()
        self.decoder_1 = Decoder_fix()
        self.decoder_2 = Decoder_fix()
        self.out5 = outconv(1024, 1)
        self.out4 = outconv(512, 1)
        self.out3 = outconv(256, 1)
        self.out2 = outconv(128, 1)
        self.out1 = outconv(64, 1)

    def forward(self, inputs):
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats = self.encoder(
            inputs
        )

        if self.mode == 1:
            out6, out7, out8, out9, out10 = self.decoder_1(
                bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats
            )
            # outputs = self.decoder_1(
            #     bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats
            # )
        elif self.mode == 2:
            out6, out7, out8, out9, out10 = self.decoder_2(
                bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats
            )
            # outputs = self.decoder_2(
            #     bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats
            # )
        else:
            raise ValueError("Unkown mode!")

        down2_featt = UpsampleDeterministic(upscale=2)(down2_feat)
        down3_featt = UpsampleDeterministic(upscale=4)(down3_feat)
        down4_featt = UpsampleDeterministic(upscale=8)(down4_feat)
        bottom_featt = UpsampleDeterministic(upscale=16)(bottom_feat)
        bottom_featt = self.out5(bottom_featt)
        down4_featt = self.out4(down4_featt)
        down3_featt = self.out3(down3_featt)
        down2_featt = self.out2(down2_featt)
        down1_featst = self.out1(down1_feats)

        return (
            out6,
            out7,
            out8,
            out9,
            out10,
            bottom_featt,
            down4_featt,
            down3_featt,
            down2_featt,
            down1_featst,
        )


if __name__ == "__main__":
    from torchstat import stat

    a = torch.ones(1, 1, 256, 256)
    model = Shared_Res_unet_cbam_insn(mode=1)
    stat(model, (1, 256, 256))
