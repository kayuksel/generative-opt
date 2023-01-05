##########################################
#Kamer Ali Yuksel linkedin.com/in/kyuksel#
##########################################

class FastCMA(object):
    def __init__(self, N, samples):
        self.samples = samples
        mu = samples // 2
        self.weights = torch.tensor([math.log(mu + 0.5)]).cuda()
        self.weights = self.weights - torch.linspace(
            start=1, end=mu, steps=mu).cuda().log()
        self.weights /= self.weights.sum()
        self.mueff = (self.weights.sum() ** 2 / (self.weights ** 2).sum()).item()
        # settings
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = 2 * (self.mueff - 2 + 1 / self.mueff) 
        self.cmu /= ((N + 2) ** 2 + 2 * self.mueff / 2)
        # variables
        self.mean = torch.zeros(N).cuda()
        self.b = torch.eye(N).cuda()
        self.d = self.b.clone()
        bd = self.b * self.d
        self.c = bd * bd.T
        self.pc = self.mean.clone()

    def step(self, step_size = 0.5):
        z = torch.randn(self.mean.size(0), self.samples).cuda()
        ss = self.mean.view(-1, 1) + step_size * self.b.matmul(self.d.matmul(z))
        f = calculate_reward(sparsemax(ss, dim=1).T, 
            valid_data[:-test_size], index[:-test_size])
        results = [{'parameters': ss.T[i], 'z': z.T[i], 
        'fitness': f.item()} for i, f in enumerate(f)]
        ranked_results = sorted(results, key=lambda x: x['fitness'])
        selected_results = ranked_results[0:self.samples//2]
        z = torch.stack([g['z'] for g in selected_results])
        g = torch.stack([g['parameters'] for g in selected_results])

        self.mean = (g * self.weights.unsqueeze(1)).sum(0)
        zmean = (z * self.weights.unsqueeze(1)).sum(0)
        self.pc *= (1 - self.cc)
        pc_cov = self.pc.unsqueeze(1) * self.pc.unsqueeze(1).T
        pc_cov = pc_cov + self.cc * (2 - self.cc) * self.c

        bdz = self.b.matmul(self.d).matmul(z.T)
        cmu_cov = bdz.matmul(self.weights.diag_embed())
        cmu_cov = cmu_cov.matmul(bdz.T)

        self.c *= (1 - self.c1 - self.cmu)
        self.c += (self.c1 * pc_cov) + (self.cmu * cmu_cov)
        self.d, self.b = torch.linalg.eigh(self.c, UPLO='U')
        self.d = self.d.sqrt().diag_embed()
        return ranked_results

best_reward = None
with torch.no_grad():
    cma_es = FastCMA(N = len(assets), samples=args.batch)
    for epoch in range(args.iter):
        try:
            res = cma_es.step()
        except Exception as e: 
            print(e)
            break
        weights = sparsemax(res[0]['parameters'], dim=0)
        r = calculate_reward(weights.unsqueeze(0), 
            valid_data[-test_size:], index[-test_size:])
        if best_reward is None: best_reward = r
        if r < best_reward:
            best_reward = r
            print('epoch: %i v_loss: %f' % (epoch, best_reward))
            bw = weights.detach().cpu().numpy()
            bw = pd.DataFrame(bw).set_index([assets])
            bw = bw.loc[~(bw==0).all(axis=1)]
            bw = bw.reindex(bw[0].abs().sort_values(ascending=False).index)
            bw.to_csv('best_weights.csv', header=False)
