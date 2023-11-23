import torch.nn as nn
from .stm_encoder_decoder import *

class SSDNet(nn.Module):
    """
    Basenet for stereo image disparity estimation.
    """
    def __init__(self, num_layers = 18, pretrained = True, max_disp = 192):
        super(SSDNet, self).__init__()
        self.encoder = STMDepthEncoder(num_layers = num_layers, pretrained = pretrained)
        self.decoder = STMDepthDecoder(self.encoder.num_ch_enc, max_disp)

    def init_weights(self):
        pass

    def forward(self, input_images):
        """
        Args:
            input_images(list): left and right images for Siamese Network
            init_disps(list):  left and right init disparity map for Siamese Network
        """
        assert len(input_images) == 2, 'Only accept image pairs'
        
        left_features = self.encoder(input_images[0])
        left_outputs = self.decoder(left_features)

        right_features = self.encoder(input_images[1])
        right_outputs = self.decoder(right_features)
        
        return left_outputs, right_outputs


    def functional_forward(self, input_images, params):
        """
        Args:
            input_images(list): left and right images for Siamese Network
            init_disps(list):  left and right init depth map for Siamese Network
        """
        assert len(input_images) == 2, 'Only accept image pairs'
        
        left_features = self.encoder.functional_forward(input_images[0], params)
        left_outputs = self.decoder.functional_forward(left_features, params)

        right_features = self.encoder.functional_forward(input_images[1], params)
        right_outputs = self.decoder.functional_forward(right_features, params)
        
        if self.training:
            return left_outputs, right_outputs
        else:
            return left_outputs[0], right_outputs[0]
