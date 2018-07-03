import torch
from torch import nn
from .graphnet import GraphModule
from .graphnet import Identity
from .graphnet import GlobalAvgPool
from .graphnet import Block
from .graphnet import AVG
from .graphnet import CAT
from .graphnet import get_module

def add_cell(net, name, stride=1, in_channels=128, out_channels=256, from_='cell_0,end'):
    net.add_edge(from_, name+',start')
    net.add_node(name+',start', Identity())
    # log the stride as an attribute of start node
    net.graph.nodes[name+',start']['stride'] = stride

    op0 = get_module('sepcbr_k,7', in_channels, out_channels, stride=stride)

    if stride == 1 and in_channels != 3:
        op1 = get_module('max_k,3', stride=1)
    else:
        op1 = get_module('sepcbr_k,3', in_channels, out_channels, stride=stride)

    net.add_edge(name+',start', name+',block_0')
    net.add_node(name+',block_0', Block([op0, op1]))

    net.add_edge(name+',block_0', name+',end')
    net.add_node(name+',end', CAT())


def add_cells(g, N=4, F=216, level=0, from_cell='start', from_cid=0):
    for i in range(from_cid, from_cid+N):
        cname = 'cell_{}'.format(i)
        add_cell(g, cname, stride=1, in_channels=F*2**level, out_channels=F*2**level, from_=from_cell)
        from_ = cname


def PNASNet(N=6, F=44, n_heads=1):
    cid = 0
    level = 0
    net = GraphModule()
    net.add_node('start', module=Identity())

    add_cell(net, 'cell_{}'.format(cid),
        stride=1,
        in_channels=3,
        out_channels=F*2**level,
        from_='start')
    cid += 1

    # N-1 x stride1 cell
    add_cells(net, N=N-1, F=F, level=level, from_cell='cell_{},end'.format(cid-1), from_cid=cid)
    cid += N-1

    # stride2 cell
    add_cell(net, 'cell_{}'.format(cid),
        stride=2,
        in_channels=F*2**level,
        out_channels=F*2**(level+1),
        from_='cell_{},end'.format(cid-1))
    cid += 1
    level += 1

    # N x stride1 cell
    add_cells(net, N=N, F=F, level=level, from_cell='cell_{},end'.format(cid-1), from_cid=cid)
    cid += N

    # stride2 cell
    add_cell(net, 'cell_{}'.format(cid),
        stride=2,
        in_channels=F*2**level,
        out_channels=F*2**(level+1),
        from_='cell_{},end'.format(cid-1))
    cid += 1
    level += 1

    if n_heads == 2:
        net.add_edge('cell_{},end'.format(cid), 'fc_aux')
        net.add_node('fc_aux', module=nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(F*4, 10)
        ))
        net.add_edge('fc_aux', 'end')

    # N x stride1 cell
    add_cells(net, N=N, F=F, level=level, from_cell='cell_{},end'.format(cid-1), from_cid=cid)
    cid += N

    net.add_edge('cell_{},end'.format(cid-1), 'pool')
    net.add_node('pool', module=GlobalAvgPool())
    net.add_edge('pool', 'fc')
    net.add_node('fc', module=nn.Linear(F*4, 10))
    net.add_edge('fc', 'end')
    net.add_node('end', module=Identity())
    return net

if __name__ == '__main__':
    net = PNASNet(N=6, F=44)
    x = torch.randn(256,3,32,32)

    net.cuda()
    x = x.cuda()

    out = net(x)
    print(out)

    import torch.onnx
    torch.onnx.export(net, x, "pnasnet_cifar10.onnx", verbose=False)



