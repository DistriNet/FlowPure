import torchcfm
import torch
import os
import argparse
import attacks.batch_attacks as attacks
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["CIFAR10", "CIFAR100"], required=True)
    parser.add_argument("--noise_type", type=str, choices=["pgd", "cw", "gauss"], required=True)

    args = parser.parse_args()
    dataset = args.dataset
    noise_type = args.noise_type

    FM = torchcfm.ConditionalFlowMatcher(sigma=0)
    snapshot_freq = 20000
    max_steps = 300001
    batch_size = 64
    device = "cuda"
    sigma = 0.3

    dataloader = utils.get_training_set(dataset, batch_size)
    net_model = utils.get_model('new')
    if dataset == 'CIFAR10':
        ckpt_name = 'checkpoint_c10'
        victim = utils.get_model('wrn_c10')
    else:
        ckpt_name = 'checkpoint_c100'
        victim = utils.get_model('wrn_c100')

    if noise_type == 'pgd':
        atk = attacks.PGD(victim)
    elif noise_type == 'cw':
        atk = attacks.batched_CW(victim)

    optim = torch.optim.Adam(net_model.parameters(), lr=2e-4)

    step = 0
    while step <= max_steps:
        for i, (X, y) in enumerate(dataloader):
            net_model.train()
            x1 = X.to(device)

            if noise_type == 'cw':
                c = (torch.rand(y.size())*2).view(y.size(0), 1, 1, 1).to(device)
                kappa=(torch.rand(y.size())).view(y.size(0), 1, 1, 1).to(device)
                steps = torch.randint_like(kappa, 1, 50).to(device)
                lr = torch.rand_like(steps).to(device) * 0.02
                x0 = atk(X, y, c=c, kappa=kappa, steps=steps, lr=lr).to(device)
            elif noise_type == 'pgd':
                epsilon = (torch.rand(y.size()) * 0.05).view(y.size(0), 1, 1, 1).to(device)
                x0 = atk(X, y, eps=epsilon)
            elif noise_type == 'gauss':
                x0 = x1 + sigma * torch.randn_like(x1)

            x0 = x0.clone().detach()
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = net_model(t, xt)
            loss = torch.mean((vt - ut) ** 2)

            optim.zero_grad()
            loss.backward()
            optim.step()
            print("step: {}, loss: {}".format(step, loss.item()))

            if step % snapshot_freq == 0:
                print("SAVING")
                states = [
                    net_model.state_dict(),
                    optim.state_dict(),
                ]
                os.makedirs(f'resources/checkpoints/flowpure_{noise_type}', exist_ok=True)
                torch.save(states, os.path.join(f'resources/checkpoints/flowpure_{noise_type}/{ckpt_name}_{step}.pth'))
                torch.save(states, os.path.join(f'resources/checkpoints/flowpure_{noise_type}/{ckpt_name}.pth'))
            step += 1

