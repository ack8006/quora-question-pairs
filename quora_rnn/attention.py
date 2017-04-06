import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    """Simple inner product attention model over sentences.

    Inputs: 
      xs   database - vectors to attend over, as a Tensor
             Batch x Seq_Len x Hidden_State
      q    query vector(s)
             Batch x Hidden_State
             
    Outputs:
      o    attention combination
             Batch x Hidden_State
      w    attention weights (for visualization)
             Batch x Seq_Len"""

    def __init__(self, debug=False):
        super(AttentionModel, self).__init__()
        # Softmax computes softmax on dim1 only.
        self.softmax = nn.Softmax()
        self.debug = debug

    def forward(self, xs, q):
        # Compatibilities.
        # (B, S)
        wl = self.get_scores(xs, q)
        w = self.softmax(wl)

        if self.debug:
            assert (w.sum(1) == 1).prod().data[0] == 1, \
                'softmax for all minibatches must sum to 1'

        # Weighted average.
        # (B,S) * (B,S,H) ...=> (B,1,H)
        # (B,1,S) * (B,S,H)  => (B,1,H) => (B, H)
        o = torch.bmm(w.unsqueeze(1), xs).squeeze()

        return (o, w)

    def get_scores(self, xs, q):
        """Given database and query vector, output compatibility scores.

        This class implements straight dot product. Can be overridden with
        other methods.
        
        Scores are real values [-inf, inf]. The return value will be softmaxed
        by this class to get weights for the weighted attentioned vector."""
        #      (B,S,H) * (B,H) ...=> (B,S)
        # bmm: (B,S,H) * (B,H,1)  => (B,S,1) => (B,S)
        wl = torch.bmm(xs, q.unsqueeze(2)).squeeze()


def BilinearAttentionModel(AttentionModel):
    """Attention model that uses a 'bilinear transform' for compatibility.

    For a pair (x, y), instead of computing an inner product <x, y>, do a
    quadratic form x^T*A*y where A is a learned parameter.
    
    This model needs to know the hidden layer size, unlike inprod Attention."""

    def __init__(self, hidden_size, debug=False)
        super(BilinearAttentionModel, self).__init__(debug)
        self.A = nn.Linear(hidden_size, hidden_size)

    def get_scores(self, xs, q):
        # (B,S,H) * [H,H](B,H)
        return super(BilinearAttentionModel, self).get_scores(xs, self.A(q))

