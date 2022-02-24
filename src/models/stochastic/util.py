import torch
from src.models.stochastic.nn.lstm_encoder import LSTMEncoder
from src.models.stochastic.nn.rcnn_encoder import RCNNEncoder


def get_encoder(layer, in_features, hidden_size, bidirectional=True):
    """Returns the requested layer."""
    if layer == "lstm":
        return LSTMEncoder(in_features, hidden_size,
                           bidirectional=bidirectional)
    elif layer == "rcnn":
        return RCNNEncoder(in_features, hidden_size,
                           bidirectional=bidirectional)
    else:
        raise ValueError("Unknown layer")


def get_z_stats(z=None, mask=None):
    """
    Computes statistics about how many zs are
    exactly 0, continuous (between 0 and 1), or exactly 1.
    :param z:
    :param mask: mask in [B, T]
    :return:
    """

    z = torch.where(mask, z, z.new_full([1], 1e2))

    num_0 = (z == 0.).sum().item()
    num_c = ((z > 0.) & (z < 1.)).sum().item()
    num_1 = (z == 1.).sum().item()

    total = num_0 + num_c + num_1
    mask_total = mask.sum().item()

    # assert total == mask_total, "total mismatch"
    return num_0, num_c, num_1, mask_total