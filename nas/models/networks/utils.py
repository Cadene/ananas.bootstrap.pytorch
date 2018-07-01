

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def memory_usage(device=0):
#     import gpustat
#     gpu_stats = gpustat.GPUStatCollection.new_query()
#     item = gpu_stats.jsonify()["gpus"][device]
#     return item