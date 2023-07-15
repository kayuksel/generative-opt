import torch
import torch.nn as nn
import torch.optim as optim
torch.set_printoptions(precision=10)

weights = nn.Parameter(torch.zeros((args.batch, len(assets)), dtype=torch.float32).to(device))

# List of optimizers available in PyTorch
optimizers = [optim.SGD, optim.Adam, optim.AdamW, optim.Adamax, optim.ASGD, optim.NAdam, 
    optim.Adagrad, optim.Adadelta, optim.Rprop, optim.RMSprop, optim.RAdam, optim.LBFGS]

def closure():
    opt.zero_grad()
    loss = calculate_reward(weights, valid_data[:-test_size], index[:-test_size], True)
    loss.mean().backward()
    return loss.mean()

for optimizer in optimizers:
    # Initialize weights to zeros using PyTorch
    weights.data.zero_()
    
    # Create an instance of the current optimizer
    opt = optimizer([weights], lr = 1e-3)

    best_weights = None
    best_loss = float("inf")

    if optimizer == optim.LBFGS:
        opt = optimizer([weights], lr=1e-3, max_iter=args.iter)
        opt.step(closure)
    else:
        opt = optimizer([weights], lr=1e-3)
        for epoch in range(args.iter):
            # Forward pass
            loss = calculate_reward(weights, valid_data[:-test_size], index[:-test_size], True)
            
            # Store the best weights
            if loss.min().item() < best_loss:
                best_loss = loss.min().item()
                best_weights = weights.clone().detach()
            
            # Backward pass and optimization
            opt.zero_grad()
            loss.mean().backward()
            opt.step()

    with torch.no_grad():
        # Calculate the test loss using the best weights
        test_loss = calculate_reward(weights[loss.argmin()].unsqueeze(0),valid_data[-test_size:], index[-test_size:])[0]
        print('%s %f' % (optimizer.__name__, test_loss))