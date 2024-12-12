import torch
from model import utility_fn

def utility_per_link(X, N):
    numerators = X.diagonal(dim1=1, dim2=2)  
    denominators = torch.sum(X, dim=2) - numerators
    fraction = numerators / (N + denominators)
    fraction = torch.log2(1+fraction)

    return fraction


def gready_scheduler(zs, X, H, N, use_utility = True):

    batch_size = zs.size()[0]
    num_var = zs.size()[1]
    schedule = torch.zeros((batch_size, num_var), dtype=torch.float64)

    sinrs = utility_per_link(X, N)
    score_prob = torch.einsum("ij, ij -> ij",zs , sinrs)
    sorted_links = torch.argsort(score_prob, descending=True, dim=1)

    

    for i in range(batch_size):
        Xi = X[i]
        Hi = H[i]
        sorted_linksi = sorted_links[i]
        schedulei = torch.zeros(num_var, dtype=torch.float64)
        utility = 0
        for link in sorted_linksi:
            schedule_step = schedulei.clone()
            schedule_step[link] = 1
            flag = True
            for hyperedge in Hi:
                non_zero_idx = torch.nonzero(hyperedge).squeeze(0)
                if torch.sum(schedule_step[non_zero_idx]) == len(non_zero_idx):
                    flag = False
                    break
            
            utility_step = utility_fn(schedule_step.unsqueeze(0), Xi.unsqueeze(0), N)   
            
            if use_utility:
                if (utility_step > utility) and flag:
                    utility = utility_step
                    schedulei = schedule_step
            else:
                if flag:
                    schedulei = schedule_step
            
        schedule[i] = schedulei.clone()
            

    return schedule