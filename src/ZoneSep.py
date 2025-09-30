import torch
import torch as th
import torch.nn as nn
from torchaudio.transforms import Spectrogram, InverseSpectrogram
from utils.tensor import ComplexTensor


class FeatureExtract(nn.Module):
    """
    creat features : Magnitude and STFT Feature
    """
    def __init__(self, channels=4, win_len=512, fft_len=512, win_inc=256):
        super(FeatureExtract, self).__init__()
        self.win_inc = win_inc
        self.stft = Spectrogram(n_fft=fft_len, win_length=win_len, hop_length=win_inc, power=None)

    def forward(self, inputs):
        """
        pad_len = self.win_inc - (inputs.shape[-1] % self.win_inc)
        inputs = torch.nn.functional.pad(inputs, (0, pad_len))
        """
        wav_stft = self.stft(inputs)
        mag = torch.abs(wav_stft)

        return mag, wav_stft


def einsum(equation, operands):
    """Einsum

    >>> import numpy
    >>> def get(*shape):
    ...     real = numpy.random.rand(*shape)
    ...     imag = numpy.random.rand(*shape)
    ...     return real + 1j * imag
    >>> x = get(3, 4, 5)
    >>> y = get(3, 5, 6)
    >>> z = get(3, 6, 7)
    >>> test = einsum('aij,ajk,akl->ail',
    ...               [ComplexTensor(x), ComplexTensor(y), ComplexTensor(z)])
    >>> valid = numpy.einsum('aij,ajk,akl->ail', x, y, z)
    >>> numpy.testing.assert_allclose(test.numpy(), valid)

    """
    x = operands[0]
    if isinstance(x, ComplexTensor):
        real_operands = [[x.real]]
        imag_operands = [[x.imag]]
    else:
        real_operands = [[x]]
        imag_operands = []

    for x in operands[1:]:
        if isinstance(x, ComplexTensor):
            real_operands, imag_operands = \
                [ops + [x.real] for ops in real_operands] + \
                [ops + [-x.imag] for ops in imag_operands], \
                [ops + [x.imag] for ops in real_operands] + \
                [ops + [x.real] for ops in imag_operands]
        else:
            real_operands = [ops + [x] for ops in real_operands]
            imag_operands = [ops + [x] for ops in imag_operands]

    real = sum([torch.einsum(equation, ops) for ops in real_operands])
    imag = sum([torch.einsum(equation, ops) for ops in imag_operands])
    return ComplexTensor(real, imag)


def get_power_spectral_density_matrix_self_with_cm_t(xs):
    # xs.shape: [B,F,C,T]
    psd = einsum('...ct,...et->...tce', [xs, xs.conj()]) # [...,T,C,E]: 对每个时频点（TF bin）均计算一个[C,E]维的协方差矩阵
    return psd


def apply_beamforming_vector(beamform_vector, mix):
    es = einsum('baftc,bafct->baft', [beamform_vector.conj(), mix]) # 将模型估计的bf参数([B,area,F,T,C])同经过复制的多通道混合信号([B,area,F,C,T])相乘相加
    return es


class CTF_Atten_block(nn.Module):
    def __init__(self, hid_dim, bands, n_heads=4, dropout_rate=0.1):
        super(CTF_Atten_block, self).__init__()
        self.hid_dim = hid_dim
        self.bands = bands

        # Channel-wise attention
        self.norm_c = nn.LayerNorm(bands)
        self.attn_c = nn.MultiheadAttention(embed_dim=bands, num_heads=n_heads, batch_first=True)
        self.dropout_c = nn.Dropout(dropout_rate)

        # Temporal attention
        self.norm_t = nn.LayerNorm(hid_dim)
        self.attn_t = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=n_heads, batch_first=True)
        self.dropout_t = nn.Dropout(dropout_rate)

        # Frequency attention
        self.norm_f = nn.LayerNorm(hid_dim)
        self.attn_f = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=n_heads, batch_first=True)
        self.dropout_f = nn.Dropout(dropout_rate)

    def forward(self, ipt):
        B, _, _, T = ipt.shape

        # 1. Channel Attention: [B*T, E, D]
        x = ipt.permute(0, 3, 1, 2).reshape(B * T, self.hid_dim, self.bands)  # [B*T, E, D]
        x_norm = self.norm_c(x)
        attn_out, _ = self.attn_c(x_norm, x_norm, x_norm)
        x = x + self.dropout_c(attn_out)

        # 2. Temporal Attention: [B*D, T, E]
        x = x.reshape(B, T, self.hid_dim, self.bands).permute(0, 3, 1, 2).reshape(B * self.bands, T, self.hid_dim)
        x_norm = self.norm_t(x)
        attn_out, _ = self.attn_t(x_norm, x_norm, x_norm)
        x = x + self.dropout_t(attn_out)

        # 3. Frequency Attention: [B*T, D, E]
        x = x.reshape(B, self.bands, T, self.hid_dim).transpose(1, 2).reshape(B * T, self.bands, self.hid_dim)
        x_norm = self.norm_f(x)
        attn_out, _ = self.attn_f(x_norm, x_norm, x_norm)
        x = x + self.dropout_f(attn_out)

        # Output: [B, E, D, T]
        x = x.reshape(B, T, self.bands, self.hid_dim).permute(0, 3, 2, 1)
        return x


class ZoneSep(nn.Module):
    def __init__(self,
                 fft_len=512,
                 win_inc=256,
                 win_len=512,
                 channels=4,
                 area=4,
                 dropout_rate=0.2,
                 lay_num=8,
                 ):
        super(ZoneSep, self).__init__()
        self.dropout_rate = dropout_rate
        self.fft_len = fft_len
        self.channels = channels
        self.areas = area

        self.stft = Spectrogram(n_fft=fft_len, hop_length=win_inc, win_length=win_len, power=None)
        self.feature = FeatureExtract(channels=channels, win_len=win_len, fft_len=fft_len, win_inc=win_inc)
        self.inverse_stft = InverseSpectrogram(n_fft=fft_len, hop_length=win_inc, win_length=win_len)

        self.hid_dim = 64
        self.out_channel = 160
        self.bands = 80

        # Encoder
        self.feat_emb_1 = nn.Conv1d(36, self.hid_dim, 1)  # nn.Conv1d(36, self.hid_dim, 3, padding='same')
        self.feat_emb_norm_1 = nn.LayerNorm(self.hid_dim)
        self.feat_relu_1 = nn.PReLU()
        self.fre_emb_1 = nn.Conv1d(self.fft_len//2+1, self.out_channel, 1)
        self.fre_emb_norm_1 = nn.LayerNorm(self.out_channel)
        self.fre_emb_relu_1 = nn.PReLU()
        self.fre_emb_2 = nn.Conv1d(self.out_channel, self.bands, 1)
        self.fre_emb_norm_2 = nn.LayerNorm(self.bands)
        self.fre_emb_relu_2 = nn.PReLU()

        # STF-Attention
        atten_layer = []
        for i in range(lay_num):
            atten_layer.append(CTF_Atten_block(hid_dim=self.hid_dim, bands=self.bands))
        self.layers = nn.ModuleList(atten_layer)

        # Beamformer Decoder
        self.i_fre_emb_1 = nn.Conv1d(self.bands, self.out_channel, 1)
        self.i_fre_emb_norm_1 = nn.LayerNorm(self.out_channel)
        self.i_fre_emb_relu_1 = nn.PReLU()
        self.i_fre_emb_2 = nn.Conv1d(self.out_channel, self.fft_len//2+1, 1)
        self.i_fre_emb_norm_2 = nn.LayerNorm(self.fft_len//2+1)
        self.i_fre_emb_relu_2 = nn.PReLU()

        self.get_ws_area_1 = nn.Conv1d(self.hid_dim, 2 * channels, 1)
        self.get_ws_area_2 = nn.Conv1d(self.hid_dim, 2 * channels, 1)
        self.get_ws_area_3 = nn.Conv1d(self.hid_dim, 2 * channels, 1)
        self.get_ws_area_4 = nn.Conv1d(self.hid_dim, 2 * channels, 1)

        # Post-mask Decoder
        self.mask_i_fre_emb_1 = nn.Conv1d(self.bands, self.out_channel, 1)
        self.mask_i_fre_emb_norm_1 = nn.LayerNorm(self.out_channel)
        self.mask_i_fre_emb_relu_1 = nn.PReLU()
        self.mask_i_fre_emb_2 = nn.Conv1d(self.out_channel, self.fft_len//2+1, 1)
        self.mask_i_fre_emb_norm_2 = nn.LayerNorm(self.fft_len//2+1)
        self.mask_i_fre_emb_relu_2 = nn.PReLU()

        self.get_mask_area_1 = nn.Conv1d(self.hid_dim, 1, 1)
        self.get_mask_area_2 = nn.Conv1d(self.hid_dim, 1, 1)
        self.get_mask_area_3 = nn.Conv1d(self.hid_dim, 1, 1)
        self.get_mask_area_4 = nn.Conv1d(self.hid_dim, 1, 1)

    def forward(self, input):
        ###### feature extraction
        mag, stft = self.feature(input) # mag: real([B,ch,F,T]); stft: cplx([B,ch,F,T])
        B, C, F, T = stft.shape
        stft_cplx = ComplexTensor(stft.real, stft.imag) # [B,ch,F,T]
        phi_yy_cplx = get_power_spectral_density_matrix_self_with_cm_t(stft_cplx.transpose(1, 2)).flatten(-2) # [B,F,T,ch]
        phi_yy = torch.cat([phi_yy_cplx.real, phi_yy_cplx.imag], dim=-1) # [B,F,T,C(32)]

        ###### Encoder
        feat_fusion = torch.cat([mag, phi_yy.permute(0, 3, 1, 2)], dim=1)  # [mag, phi]
        feat_fusion = feat_fusion.transpose(1, 2).reshape(B * F, -1, T) # [B*F,C=36,T]
        feat_emb = self.feat_emb_1(feat_fusion) # 空间特征提取(36->64)：[B*F,E=128,T]
        feat_emb = self.feat_relu_1(self.feat_emb_norm_1(feat_emb.transpose(-1, -2))).transpose(-1, -2) # [B*F,E(64),T]
        feat_emb = feat_emb.reshape(B, F, self.hid_dim, T).transpose(1, 2).reshape(B*self.hid_dim, F, T) # [B*E(64),F(257),T]
        feat_emb = self.fre_emb_1(feat_emb) # 频率压缩1(257->160)：[B*E(64),D(160),T]
        feat_emb = self.fre_emb_relu_1(self.fre_emb_norm_1(feat_emb.transpose(-1, -2))).transpose(-1, -2) # [B*E(64),D(160),T]
        ws_emb = self.fre_emb_2(feat_emb).reshape(B, self.hid_dim, self.bands, T) # 频率压缩2(160->80)：[B*E(64),D(80),T]
        ws_emb = self.fre_emb_relu_2(self.fre_emb_norm_2(ws_emb.transpose(-1, -2))).transpose(-1, -2) 

        ###### Separator
        for m in self.layers:
            ws_emb = m(ws_emb) # 空间全局建模：[B*E(64),D(80),T]

        ###### Neural Beamforming Decoder
        ws_emb_bf =  ws_emb
        ws_emb_bf = ws_emb_bf.reshape(B*self.hid_dim, self.bands, T) # [B*E(64),D(80),T]
        ws_emb_bf = self.i_fre_emb_relu_1(self.i_fre_emb_norm_1(self.i_fre_emb_1(ws_emb_bf).transpose(-1, -2))).transpose(-1, -2) # 频率还原1(80->160): [B*E(64),D(160),T] # TODO: skip-connection ?
        ws_emb_bf = self.i_fre_emb_relu_2(self.i_fre_emb_norm_2(self.i_fre_emb_2(ws_emb_bf).transpose(-1, -2))).transpose(-1, -2) # 频率还原2(160->257): [B*E(64),F(257),T] # TODO: skip-connection ?
        ws_emb_bf = ws_emb_bf.reshape(B, self.hid_dim, F, T).transpose(1, 2).reshape(B*F, self.hid_dim, T) # [B*F(257),E(64),T]
        ws_area1 = self.get_ws_area_1(ws_emb_bf).reshape(B, F, self.channels * 2, T) # 音区1的波束形成参数：[B,F(257),C(8),T]
        ws_area2 = self.get_ws_area_2(ws_emb_bf).reshape(B, F, self.channels * 2, T) # 音区2的波束形成参数：[B,F(257),C(8),T]
        ws_area3 = self.get_ws_area_3(ws_emb_bf).reshape(B, F, self.channels * 2, T) # 音区3的波束形成参数：[B,F(257),C(8),T]
        ws_area4 = self.get_ws_area_4(ws_emb_bf).reshape(B, F, self.channels * 2, T) # 音区4的波束形成参数：[B,F(257),C(8),T]

        ###### Mask Decoder
        ws_emb_mask = ws_emb
        ws_emb_mask = ws_emb_mask.reshape(B*self.hid_dim, self.bands, T) # [B*E(64),D(80),T]
        ws_emb_mask = self.mask_i_fre_emb_relu_1(self.mask_i_fre_emb_norm_1(self.mask_i_fre_emb_1(ws_emb_mask).transpose(-1, -2))).transpose(-1, -2) # 频率还原1(80->160): [B*E(64),D(160),T] # TODO: skip-connection ?
        ws_emb_mask = self.mask_i_fre_emb_relu_2(self.mask_i_fre_emb_norm_2(self.mask_i_fre_emb_2(ws_emb_mask).transpose(-1, -2))).transpose(-1, -2) # 频率还原2(160->257): [B*E(64),F(257),T] # TODO: skip-connection ?
        ws_emb_mask = ws_emb_mask.reshape(B, self.hid_dim, F, T).transpose(1, 2).reshape(B*F, self.hid_dim, T) # [B*F(257),E(64),T]
        mask_area1 = self.get_mask_area_1(ws_emb_mask).reshape(B, F, 1, T) # 音区1的幅度谱mask：[B,F(257),ch(1),T]
        mask_area2 = self.get_mask_area_2(ws_emb_mask).reshape(B, F, 1, T) # 音区2的幅度谱mask：[B,F(257),ch(1),T]
        mask_area3 = self.get_mask_area_3(ws_emb_mask).reshape(B, F, 1, T) # 音区3的幅度谱mask：[B,F(257),ch(1),T]
        mask_area4 = self.get_mask_area_4(ws_emb_mask).reshape(B, F, 1, T) # 音区4的幅度谱mask：[B,F(257),ch(1),T]

        ###### Apply Beamforming
        ws_mix = torch.stack([ws_area1, ws_area2, ws_area3, ws_area4], dim=2).transpose(1, 2) # [B,F(257),a(4),C(8),T] -> [B,a(4),F,C(8),T]
        ws_total_cplx = ComplexTensor(ws_mix[..., :self.channels, :], ws_mix[..., self.channels:, :]).transpose(-1, -2) # [B,a(4),F,C(4),T] -> [B,a(4),F,T,C(4)]
        enhanced_spec_per_area_cplx = apply_beamforming_vector(ws_total_cplx, stft_cplx.transpose(1, 2).unsqueeze(1).repeat(1, self.areas, 1, 1, 1)) # [B,a(4),F,T]
        #### ISTFT 
        mid_enhanced_spec_per_area = enhanced_spec_per_area_cplx.real + 1j * enhanced_spec_per_area_cplx.imag # [B,a,F,T]
        mid_result_per_area = self.inverse_stft(mid_enhanced_spec_per_area) # [B,a,T]
       
        ###### Apply Mask
        enhanced_spec_per_area_cplx_detached = enhanced_spec_per_area_cplx.detach()
        enhanced_mag_per_area = torch.sqrt(enhanced_spec_per_area_cplx_detached.real**2 + enhanced_spec_per_area_cplx_detached.imag**2)  # [B,ch,F,T]
        phase_mix = torch.atan2(enhanced_spec_per_area_cplx_detached.imag, enhanced_spec_per_area_cplx_detached.real)  # [B,ch(4),F(257),T]
        mask_mix = torch.cat([mask_area1, mask_area2, mask_area3, mask_area4], dim=2).transpose(1, 2) # [B,ch(4),F(257),T]
        mag_mix = mask_mix * enhanced_mag_per_area # [B,ch(4),F(257),T]
        #### ISTFT
        final_enhanced_spec_per_area = (mag_mix*torch.cos(phase_mix)) + 1j * (mag_mix*torch.sin(phase_mix)) # [B,ch(4),F(257),T]
        final_result_per_area = self.inverse_stft(final_enhanced_spec_per_area) # [B,ch,T]

        return mid_result_per_area, mid_enhanced_spec_per_area, final_result_per_area, final_enhanced_spec_per_area


def testing():
    model = ZoneSep(fft_len=512, win_inc=256, win_len=512, channels=4).eval()

    """calc time"""
    import time
    noisy = th.rand(1, 4, 16000) # [B,C,T]
    start_time = time.time()
    y = model(noisy)[0]
    end_time = time.time()
    run_time = end_time - start_time
    print(f'inference time:{run_time:.3f} s.')

    """complexity count using <ptflops.get_model_complexity_info>"""
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (4, 16000), as_strings=True,
                                        print_per_layer_stat=False, verbose=False)
    print(f"<ptflops.get_model_complexity_info>\nMACs:{macs}, params:{params}")


if __name__ == "__main__":
    testing()
