from flax import linen as nn

class RSSM(nn.Module):
    def setup(self):
        self.deterministicRNN = nn.GRUCell(name="deterministic_rnn")
        self.stochasticRNN = None

    @nn.compact
    def __call__(self, x, h, c):
        x, _ = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        h, c = self.rnn(x, (h, c))
        x = self.dense(h)
        return x, h, c
