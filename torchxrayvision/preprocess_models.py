import torch.nn as nn
import pdb

import sys
sys.path.append('/lotterlab/lotterb/repos/nflows/')
from nflows.transforms.splines import rational_quadratic_spline

class MonotonicSplineWindower(nn.Module):
    def __init__(self,  params_model, n_bins, n_pixels=256, predict_model=None):
        super().__init__()
        self.n_bins = n_bins
        self.params_model = params_model
        self.predict_model = predict_model
        self.n_pixels = n_pixels

    def forward(self, x):
        params = self.params_model(x)
        widths = params[:, :self.n_bins]
        heights = params[:, self.n_bins:2 * self.n_bins]
        derivatives = params[:, 2 * self.n_bins:]

        orig_shape = x.shape
        x_flat = x.view(orig_shape[0], -1)
        #pdb.set_trace()
        x_windowed, _ = rational_quadratic_spline(x_flat, widths, heights, derivatives,
                                                  right=self.n_pixels, top=self.n_pixels)
        x_windowed = x_windowed.view(*orig_shape)
        if self.predict_model is not None:
            return self.predict_model(x_windowed)
        else:
            return x_windowed




