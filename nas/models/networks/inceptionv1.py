from torch import nn
from .graphnet import CBR
from .graphnet import Identity
from .graphnet import GraphModule

def GraphInceptionV1(out_channels=160): #96, 128, 160
    def add_cell(net, name, stride=1, in_channels=128, out_channels=256, from_='cell_0.end'):
        net.add_edge(from_, name+'.start')
        net.add_node(name+'.start', Identity())
        # log the stride as an attribute of start node
        net.graph.nodes[name+'.start']['stride'] = stride

        # branch1x1
        net.add_edge(name+'.start', name+'.block_0')
        net.add_node(name+'.block_0', Block([
            CBR(in_channels, out_channels, 1, stride=stride, padding=0),
            CBR(in_channels, out_channels, 1, stride=stride, padding=0),
        ]))

        # branch3x3
        net.add_edge(name+'.start', name+'.block_1')
        net.add_node(name+'.block_1', Block([
            CBR(in_channels, out_channels, 1, stride=stride, padding=0),
            CBR(in_channels, out_channels, 1, stride=stride, padding=0),
        ]))
        net.add_edge(name+'.block_1', name+'.block_2')
        net.add_node(name+'.block_2', Block([
            CBR(out_channels, out_channels, 3, stride=stride, padding=1),
            CBR(out_channels, out_channels, 3, stride=stride, padding=1),
        ]))

        # branch5x5
        net.add_edge(name+'.start', name+'.block_3')
        net.add_node(name+'.block_3', Block([
            CBR(in_channels, out_channels, 1, stride=stride, padding=0),
            CBR(in_channels, out_channels, 1, stride=stride, padding=0),
        ]))
        net.add_edge(name+'.block_3', name+'.block_4')
        net.add_node(name+'.block_4', Block([
            CBR(out_channels, out_channels, 5, stride=stride, padding=2),
            CBR(out_channels, out_channels, 5, stride=stride, padding=2),
        ]))

        # branch_pool
        net.add_edge(name+'.start', name+'.block_5')
        net.add_node(name+'.block_5', Block([
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=1, padding=1),
        ]))
        net.add_edge(name+'.block_5', name+'.block_6')
        net.add_node(name+'.block_6', Block([
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=1, padding=1),
        ]))
        
        net.add_edge(name+'.block_0', name+'.end')
        net.add_edge(name+'.block_2', name+'.end')
        net.add_edge(name+'.block_4', name+'.end')
        net.add_edge(name+'.block_6', name+'.end')
        net.add_node(name+'.end', AVG())

    # in_shape = (4,416,640)
    net = GraphModule()
    net.add_node('start', Identity())
    net.add_edge('start', 'layer_1')
    net.add_node('layer_1', nn.Sequential(
        CBR(4, 64, 7, stride=2, padding=3),
        nn.MaxPool2d(3, stride=2, padding=1)))
    net.add_edge('layer_1', 'layer_2')
    net.add_node('layer_2', nn.Sequential(
        CBR(64, 64, 1, stride=1, padding=0),
        CBR(64, out_channels, 3, stride=1, padding=1),
        #CBR(64, 192, 3, stride=1, padding=1),
        nn.MaxPool2d(3, stride=2, padding=1)))
    # layer 3
    # (0)64+128+32+32=256
    add_cell(net, 'cell_0', stride=1, in_channels=out_channels, out_channels=out_channels, from_='layer_2')
    # (1)128+192+96+64=480
    add_cell(net, 'cell_1', stride=1, in_channels=out_channels, out_channels=out_channels, from_='cell_0.end')
    net.add_edge('cell_1.end', 'layer_3_pool')
    net.add_node('layer_3_pool', nn.MaxPool2d(3, stride=2, padding=1))
    # layer 4
    # (0)192+208+48+64=512
    add_cell(net, 'cell_2', stride=1, in_channels=out_channels, out_channels=out_channels, from_='layer_3_pool')
    # (1)160+224+64+64=512
    add_cell(net, 'cell_3', stride=1, in_channels=out_channels, out_channels=out_channels, from_='cell_2.end')
    # (2)128+256+64+64=512
    add_cell(net, 'cell_4', stride=1, in_channels=out_channels, out_channels=out_channels, from_='cell_3.end')
    # (3)112+288+64+64=528
    add_cell(net, 'cell_5', stride=1, in_channels=out_channels, out_channels=out_channels, from_='cell_4.end')
    # (4)256+320+128+128=832
    add_cell(net, 'cell_6', stride=1, in_channels=out_channels, out_channels=out_channels, from_='cell_5.end')
    # layer 5
    # (0)256+320+128+128=832
    add_cell(net, 'cell_7', stride=1, in_channels=out_channels, out_channels=out_channels, from_='cell_6.end')
    # (1)384+384+128+128=1024
    add_cell(net, 'cell_8', stride=1, in_channels=out_channels, out_channels=out_channels, from_='cell_7.end')

    # BEWARE OF THE ORDER
    net.add_edge('layer_1', 'end')
    net.add_edge('layer_2', 'end')
    net.add_edge('layer_3_pool', 'end')
    net.add_edge('cell_6.end', 'end') # layer_4
    net.add_edge('cell_8.end', 'end') # layer_5
    net.add_node('end', Identity())
    net.output_shape = (out_channels, 26, 40) # required for hydra API
    return net