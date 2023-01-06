##########################################
#Kamer Ali Yuksel linkedin.com/in/kyuksel#
##########################################

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = 5/3)
        if hasattr(m, 'bias') and m.bias is not None: m.bias.data.zero_()

class LSTMModule(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 1, num_layers = 2):
        super(LSTMModule, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.h = torch.zeros(num_layers, 1, hidden_size, requires_grad=True).to(device)
        self.c = torch.zeros(num_layers, 1, hidden_size, requires_grad=True).to(device)
    def forward(self, x):
        self.rnn.flatten_parameters()
        out, (h_end, c_end) = self.rnn(x, (self.h, self.c))
        self.h.data = h_end.data
        self.c.data = c_end.data
        return out[:,-1, :].flatten()

class Extractor(nn.Module):
    def __init__(self, latent_dim, ks = 5):
        super(Extractor, self).__init__()
        self.conv = nn.Conv1d(args.noise, latent_dim,
            bias = False, kernel_size = ks, padding = (ks // 2) + 1)
        self.conv.weight.data.normal_(0, 0.01)
        self.activation = nn.Sequential(nn.BatchNorm1d(
            latent_dim, track_running_stats = False), nn.Mish())
        self.gap = nn.AvgPool1d(kernel_size = args.batch, padding = 1)
        self.rnn = LSTMModule(hidden_size = latent_dim)
    def forward(self, x):
        y = x.unsqueeze(0).permute(0, 2, 1)
        y = self.rnn(self.gap(self.activation(self.conv(y))))
        return torch.cat([x, y.repeat(args.batch, 1)], dim = 1)

class Generator(nn.Module):
    def __init__(self, noise_dim = 0):
        super(Generator, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Tanh()]
        self.model = nn.Sequential(
            *block(noise_dim+args.cnndim, 512), *block(512, 1024), nn.Linear(1024, len(assets)))
        init_weights(self)
        self.extract = Extractor(args.cnndim)
        self.std_weight = nn.Parameter(torch.zeros(len(assets)).to(device)) # changing this for convenience of GradInit
    def forward(self, x):
        mu = self.model(self.extract(x))
        return mu, mu + (self.std_weight * torch.randn_like(mu))

actor = Generator(args.noise).to(device)
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=1e-3)

best_reward = None

for epoch in range(args.iter):
    torch.cuda.empty_cache()
    weights, dweights = actor(torch.randn((args.batch, args.noise)).to(device))
    dweights = nn.functional.dropout(dweights, p = 0.75).softmax(dim=1)
    loss = calculate_reward(dweights, valid_data[:-test_size], index[:-test_size], True).mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt.step()

    with torch.no_grad():
        #entmax15 actually is better but resulting portfolios are less sparse
        weights = sparsemax(weights.mean(dim=0), dim=0)
        test_reward = calculate_reward(weights.unsqueeze(0), 
            valid_data[-test_size:], index[-test_size:])[0]

        if best_reward is None: best_reward = test_reward
        if test_reward < best_reward:
            best_reward = test_reward
            print('epoch: %i v_loss: %f' % (epoch, best_reward))
            bw = weights.detach().cpu().numpy()
            bw = pd.DataFrame(bw).set_index([assets])
            bw = bw.loc[~(bw==0).all(axis=1)]
            bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
            bw.to_csv('best_weights.csv', header=False)
