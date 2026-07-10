import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

# ==================== FRD (Feature Recalibration Decoder) Block ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


# ==================== 核心组件 ====================
class SAM2AdapterLayer(nn.Module):
    """SAM2适配器层"""
    def __init__(self, in_channels, out_channels, embed_dim=256, depth=2, scale_factor=2):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # 输入处理
        if scale_factor > 1:
            self.input_conv = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim // 2, 3, stride=scale_factor, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.ReLU(inplace=True)
            )
        else:
            self.input_conv = nn.Conv2d(in_channels, embed_dim // 2, 3, padding=1)
        
        # 简化的注意力模块
        self.attention = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 输出转换
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 如果需要上采样
        if scale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Identity()
    
    def forward(self, x):
        # 输入处理
        x_in = self.input_conv(x)
        
        # 注意力机制
        attn = self.attention(x_in)
        x_att = x_in * attn + x_in
        
        # 输出转换
        x_out = self.output_conv(x_att)
        
        # 上采样
        x_out = self.upsample(x_out)
        
        return x_out


# ========== 基础模块 ==========
class VKANBlock(nn.Module):
    """简化的VKAN块"""

    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()

        # 确保参数是整数
        dim = int(dim)
        expanded_dim = int(dim * mlp_ratio)

        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.GELU()

        self.norm2 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, expanded_dim, 1)
        self.conv3 = nn.Conv2d(expanded_dim, dim, 1)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        # 第一层
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = x + identity

        # 第二层
        identity = x
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = x + identity

        return x


class VKANEncoder(nn.Module):
    """VKAN编码器"""

    def __init__(self, in_channels, out_channels, num_blocks=2, mlp_ratio=4.0, drop_rate=0.0):
        super().__init__()

        # 确保通道数是整数
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 传递所有必要参数
        self.blocks = nn.Sequential(*[
            VKANBlock(
                dim=out_channels,
                mlp_ratio=mlp_ratio,
                drop=drop_rate
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class VKANDecoder(nn.Module):
    """VKAN解码器"""

    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=2, mlp_ratio=4.0, drop_rate=0.0):
        super().__init__()

        # 确保通道数是整数
        in_channels = int(in_channels)
        skip_channels = int(skip_channels)
        out_channels = int(out_channels)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if skip_channels > 0:
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(out_channels + skip_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.skip_fusion = None

        # 传递所有必要参数给VKANBlock
        self.blocks = nn.Sequential(*[
            VKANBlock(
                dim=out_channels,
                mlp_ratio=mlp_ratio,
                drop=drop_rate
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None and self.skip_fusion is not None:
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = self.skip_fusion(x)

        x = self.blocks(x)
        return x


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(combined)
        return self.sigmoid(attention)


class CBAM(nn.Module):
    """结合通道和空间注意力的CBAM模块"""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


class LearnableSkipConnection(nn.Module):
    """
    可学习跳跃连接模块
    结合通道注意力、空间注意力和可学习权重
    """

    def __init__(self, encoder_channels, decoder_channels, use_cbam=True):
        super(LearnableSkipConnection, self).__init__()

        # 对齐通道数
        self.conv_align = nn.Conv2d(encoder_channels, decoder_channels, 1)
        self.bn = nn.BatchNorm2d(decoder_channels)
        self.relu = nn.ReLU(inplace=True)

        # 注意力机制
        self.use_cbam = use_cbam
        if use_cbam:
            self.attention = CBAM(decoder_channels)

        # 可学习权重参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 编码器特征权重
        self.beta = nn.Parameter(torch.tensor(0.5))  # 解码器特征权重

    def forward(self, encoder_feat, decoder_feat):
        """
        Args:
            encoder_feat: 编码器特征 [B, C_enc, H, W]
            decoder_feat: 解码器特征 [B, C_dec, H, W]
        Returns:
            enhanced_feat: 增强后的特征 [B, C_dec, H, W]
        """
        # 1. 对齐编码器特征的通道数
        aligned_encoder = self.relu(self.bn(self.conv_align(encoder_feat)))

        # 2. 应用注意力机制
        if self.use_cbam:
            aligned_encoder = self.attention(aligned_encoder)

        # 3. 调整大小（如果空间维度不一致）
        if aligned_encoder.shape[-2:] != decoder_feat.shape[-2:]:
            aligned_encoder = F.interpolate(
                aligned_encoder,
                size=decoder_feat.shape[-2:],
                mode='bilinear',
                align_corners=True
            )

        # 4. 可学习加权融合（使用sigmoid确保权重在0-1之间）
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)

        # 归一化权重
        total = alpha + beta
        alpha = alpha / total
        beta = beta / total

        # 融合特征
        enhanced_feat = alpha * aligned_encoder + beta * decoder_feat

        return enhanced_feat


class FRDBlock(nn.Module):
    """简化的特征重校准模块（避免序列转换问题）"""

    def __init__(
            self,
            in_channels,
            skip_channels=0,
            out_channels=None,
            use_attention=True,
            reduction_ratio=16
    ):
        super().__init__()

        # 参数设置
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.skip_channels = int(skip_channels)
        self.use_attention = use_attention

        print(f"[FRDBlock] in={self.in_channels}, out={self.out_channels}, skip={self.skip_channels}")

        # 通道调整
        if self.in_channels != self.out_channels:
            self.channel_adjust = nn.Conv2d(self.in_channels, self.out_channels, 1)
        else:
            self.channel_adjust = nn.Identity()

        # 跳跃连接融合
        if self.skip_channels > 0:
            self.skip_fusion = nn.Conv2d(
                self.in_channels + self.skip_channels,
                self.in_channels,
                1
            )
        else:
            self.skip_fusion = None

        # 特征处理
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.in_channels)

        # 注意力
        if use_attention:
            self.attention = CBAM(self.in_channels, reduction_ratio)

    def forward(self, x, skip=None):
        # 处理跳跃连接
        if skip is not None and self.skip_fusion is not None:
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = self.skip_fusion(x)

        # 特征处理
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # 注意力
        if hasattr(self, 'attention'):
            x = self.attention(x)

        # 残差连接
        x = x + identity

        # 通道调整
        x = self.channel_adjust(x)

        return x


class MLSN_LSC_SAM2_VKAN_FRD(nn.Module):
    """完整的MLSN_LSC_SAM2_VKAN_FRD网络"""

    def __init__(self, n_classes=1, in_channels=3, base_channels=64, img_size=256):
        super().__init__()

        self.n_classes = n_classes
        self.base_channels = base_channels

        # 打印调试信息
        print(f"[DEBUG] 初始化 MLSN_LSC_SAM2_VKAN_FRD")
        print(f"  base_channels={base_channels}")
        print(f"  通道分布:")
        print(f"    encoder1 -> {base_channels}")
        print(f"    encoder2 -> {base_channels * 2}")
        print(f"    encoder3 -> {base_channels * 4}")
        print(f"    encoder4 -> {base_channels * 8}")
        print(f"    bottleneck -> {base_channels * 16}")

        # ========== 编码器部分 ==========
        # 第1级：SAM2增强编码器（简化版本）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 第2级：SAM2增强编码器（简化版本）
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        # 第3级：VKAN编码器
        self.encoder3 = VKANEncoder(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            num_blocks=2
        )

        # 第4级：VKAN编码器
        self.encoder4 = VKANEncoder(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
            num_blocks=2
        )

        # ========== 瓶颈层 ==========
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            VKANBlock(base_channels * 8),
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True)
        )

        # ========== 可学习跳跃连接 ==========
        print(f"\n[DEBUG] 初始化跳跃连接:")
        # skip4: 连接 encoder4 (512) -> decoder4的输入
        print(f"  skip4: encoder={base_channels * 8}, decoder={base_channels * 8}")
        self.skip4 = LearnableSkipConnection(
            encoder_channels=base_channels * 8,  # enc4输出: 512
            decoder_channels=base_channels * 8,  # decoder4输入: 512
            use_cbam=True
        )

        # skip3: 连接 encoder3 (256) -> decoder3的输入
        print(f"  skip3: encoder={base_channels * 4}, decoder={base_channels * 4}")
        self.skip3 = LearnableSkipConnection(
            encoder_channels=base_channels * 4,  # enc3输出: 256
            decoder_channels=base_channels * 4,  # decoder3输入: 256
            use_cbam=True
        )

        # skip2: 连接 encoder2 (128) -> decoder2的输入
        print(f"  skip2: encoder={base_channels * 2}, decoder={base_channels * 2}")
        self.skip2 = LearnableSkipConnection(
            encoder_channels=base_channels * 2,  # enc2输出: 128
            decoder_channels=base_channels * 2,  # decoder2输入: 128
            use_cbam=True
        )

        # skip1: 连接 encoder1 (64) -> decoder1的输入
        print(f"  skip1: encoder={base_channels}, decoder={base_channels}")
        self.skip1 = LearnableSkipConnection(
            encoder_channels=base_channels,  # enc1输出: 64
            decoder_channels=base_channels,  # decoder1输入: 64
            use_cbam=True
        )

        # ========== 解码器部分 ==========
        print(f"\n[DEBUG] 初始化解码器:")
        # 第4级解码器：VKAN解码器
        print(f"  decoder4: in={base_channels * 16}, skip={base_channels * 8}, out={base_channels * 8}")
        self.decoder4 = VKANDecoder(
            in_channels=base_channels * 16,  # 1024
            skip_channels=base_channels * 8,  # 512
            out_channels=base_channels * 8,  # 512
            num_blocks=2
        )

        # 第3级解码器：VKAN解码器
        print(f"  decoder3: in={base_channels * 8}, skip={base_channels * 4}, out={base_channels * 4}")
        self.decoder3 = VKANDecoder(
            in_channels=base_channels * 8,  # 512
            skip_channels=base_channels * 4,  # 256
            out_channels=base_channels * 4,  # 256
            num_blocks=2
        )

        # 第2级解码器：FRD解码器
        print(f"  decoder2 (FRDBlock): in={base_channels * 4}, skip={base_channels * 2}, out={base_channels * 2}")
        self.decoder2 = FRDBlock(
            in_channels=base_channels * 4,  # 256
            skip_channels=base_channels * 2,  # 128
            out_channels=base_channels * 2,  # 128
            use_attention=True
        )

        # 第1级解码器：FRD解码器
        print(f"  decoder1 (FRDBlock): in={base_channels * 2}, skip={base_channels}, out={base_channels}")
        self.decoder1 = FRDBlock(
            in_channels=base_channels * 2,  # 128
            skip_channels=base_channels,  # 64
            out_channels=base_channels,  # 64
            use_attention=True
        )

        # ========== 输出层 ==========
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, n_classes, 1)
        )

        # ========== 深度监督 ==========
        self.ds4 = nn.Conv2d(base_channels * 8, n_classes, 1)
        self.ds3 = nn.Conv2d(base_channels * 4, n_classes, 1)
        self.ds2 = nn.Conv2d(base_channels * 2, n_classes, 1)
        self.ds1 = nn.Conv2d(base_channels, n_classes, 1)

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            output: 分割结果 [B, n_classes, H, W]
        """
        # 编码路径
        enc1 = self.encoder1(x)  # [B, 64, H, W]
        enc2 = self.encoder2(enc1)  # [B, 128, H/2, W/2]
        enc3 = self.encoder3(enc2)  # [B, 256, H/4, W/4]
        enc4 = self.encoder4(enc3)  # [B, 512, H/8, W/8]

        # 瓶颈层
        bottleneck = self.bottleneck(enc4)  # [B, 1024, H/16, W/16]

        # 解码路径
        # 第4级解码
        dec4 = self.decoder4(bottleneck, enc4)  # [B, 512, H/8, W/8]

        # 第3级解码
        dec3 = self.decoder3(dec4, enc3)  # [B, 256, H/4, W/4]

        # 第2级解码（使用FRDBlock）
        dec2 = self.decoder2(dec3, enc2)  # [B, 128, H/2, W/2]

        # 第1级解码（使用FRDBlock）
        dec1 = self.decoder1(dec2, enc1)  # [B, 64, H, W]

        # 最终输出
        output = self.output(dec1)  # [B, n_classes, H, W]

        # 深度监督输出（如果需要）
        if self.training:
            ds4 = self.ds4(dec4)
            ds3 = self.ds3(dec3)
            ds2 = self.ds2(dec2)
            ds1 = self.ds1(dec1)

            # 上采样到原始尺寸
            ds4 = F.interpolate(ds4, size=x.shape[2:], mode='bilinear', align_corners=True)
            ds3 = F.interpolate(ds3, size=x.shape[2:], mode='bilinear', align_corners=True)
            ds2 = F.interpolate(ds2, size=x.shape[2:], mode='bilinear', align_corners=True)
            ds1 = F.interpolate(ds1, size=x.shape[2:], mode='bilinear', align_corners=True)

            return output, ds1, ds2, ds3, ds4

        return output
