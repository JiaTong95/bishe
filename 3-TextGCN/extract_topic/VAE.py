import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE model


class VAE(nn.Module):
    def __init__(self, encode_dims=[2000, 1024, 512, 20], decode_dims=[20, 1024, 2000], dropout=0.0):

        super(VAE, self).__init__()
        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encode_dims[i], encode_dims[i+1])
            for i in range(len(encode_dims)-2)
        })
        self.fc_mu = nn.Linear(encode_dims[-2], encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2], encode_dims[-1])

        self.decoder = nn.ModuleDict({
            f'dec_{i}': nn.Linear(decode_dims[i], decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })
        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(encode_dims[-1], encode_dims[-1])

    def encode(self, x):
        hid = x
        for i, layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, log_var = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, log_var

    def inference(self, x):
        mu, log_var = self.encode(x)
        theta = torch.softmax(x, dim=1)
        return theta

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i < len(self.decoder)-1:
                hid = F.relu(self.dropout(hid))
        return hid

    def forward(self, x, collate_fn=None):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        _theta = self.fc1(z)
        if collate_fn != None:
            theta = collate_fn(_theta)
        else:
            theta = _theta
        x_reconst = self.decode(theta)
        return x_reconst, mu, log_var, theta


if __name__ == '__main__':
    bow_dim = 555
    n_topic = 7
    model = VAE(encode_dims=[bow_dim, 512, 256, n_topic],
                decode_dims=[n_topic, 128, 768, bow_dim])
    model = model.cuda()
    print(model)
    # inpt = torch.randn(234, 1024).cuda()
    # out, mu, log_var, theta = model(inpt)
    # print(inpt, inpt.shape)
    # print(out, out.shape)
    # print(mu, mu.shape)
    # print(log_var, log_var.shape)

    inpt = torch.ones(1, bow_dim).cuda()
    out, mu, log_var, theta = model(inpt)
    print("inpt", inpt, inpt.shape)
    print("mu", mu, mu.shape)
    print("log_var", log_var, log_var.shape)
    print("theta", theta, theta.shape)
    ans = torch.softmax(theta, dim=1)
    print("ans", ans, ans.shape)
    print([round(float(x), 6) for x in ans[0]])
