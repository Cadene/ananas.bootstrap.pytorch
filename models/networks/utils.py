

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def memory_usage(device=0):
#     import gpustat
#     gpu_stats = gpustat.GPUStatCollection.new_query()
#     item = gpu_stats.jsonify()["gpus"][device]
#     return item


def count_p(state):
    n_params = 0
    n_0 = 0
    for k, p in state.items():
        if 'backbone' in k:
            n_params += p.nelement()
            n_0 += p[p==0].nelement()
    return n_params, n_0

        