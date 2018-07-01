# contact remi.cadene@icloud.com for questions

import re
import torch
import torch.nn as nn
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import collections
import operator
plt.switch_backend('agg')
from .utils import count_parameters

# https://gist.github.com/dgrant/6332309
class LastDeletedOrderedDict(dict):
    """ An ordred dictionnary that keeps the positions of the last deletions.
    Ex: d={'a':'a', 'b':'b'}
        del d['a']         # {'b':'b'}
        d['c'] = 'c'       # {'c':'c', b':'b'}
    """
    def __init__(self):
        self.order = {}
        self.count = 0
        self.dels = []

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        if key not in self:
            if len(self.dels) > 0:
                # pop() for LIFO, pop(0) for FIFO
                self.order[key] = self.dels.pop() # TODO: pop(0)?
            else:
                self.order[key] = self.count
                self.count += 1
            dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        if key in self:
            self.dels.append(self.order[key])
            del self.order[key]
            dict.__delitem__(self, key)

    def __iter__(self):
        order_items = self.order.items()
        order_items = sorted(order_items, key=operator.itemgetter(1))
        for key, order in order_items:
            yield key

    def items(self):
        for k in self:
            yield k, self[k]

    def keys(self):
        for k in self:
            yield k

    def values(self):
        for k in self:
            yield self[k]


# https://networkx.github.io/documentation/latest/_modules/networkx/classes/ordered.html
class LastDeletedOrderedDiGraph(nx.OrderedDiGraph):
    """ We need to keep track of the order in which the adjacente nodes have been added.
    Ex: A node has two consecutive predecessors: {'P0':0, 'P1':1}
        If we delete P0, we want to be sure that P1 stays at position 1: {'P1':1}
        If we then add P2, it should take the position of P0: {'P2':0, 'P1':1}
        This behavior is ensure by LastDeletedOrderdDict.
    """
    node_dict_factory = LastDeletedOrderedDict # OrderedDict only?
    adjlist_outer_dict_factory = LastDeletedOrderedDict
    adjlist_inner_dict_factory = LastDeletedOrderedDict
    edge_attr_dict_factory = LastDeletedOrderedDict # dict only?

    def fresh_copy(self):
        return LastDeletedOrderedDiGraph()


class GraphModule(nn.Module):

    def __init__(self):
        super().__init__()
        # TODO: use MultiDiGraph instead
        self.graph = LastDeletedOrderedDiGraph()

    # def state_dict(self, *args, **kwargs):
    #     # TODO: better implementation that relies on state dict of module
    #     #       instead of modules them self
    #     state = nx.node_link_data(self.graph)
    #     #import ipdb; ipdb.set_trace()
    #     return state

    # def load_state_dict(self, state, *args, **kwargs):
    #     self._modules.clear() # in case not empty
    #     self.graph = nx.node_link_graph(state)
    #     for node in state['nodes']:
    #         self.add_module(node['id'], node['module'])

    def add_node(self, name, module):
        self.graph.add_node(name, module=module)
        # makes self to be aware of its modules
        # avoid reimplementing nn.Module methods (state_dict(), cuda(), etc.)
        self.add_module(name, module)

    def remove_node(self, name):
        """ Removes a node and its edges from the graph, and
            removes it from the modules dictionnary as well.
        """
        if name not in self._modules:
            raise ValueError()
        for n in list(self.graph.predecessors(name)):
            self.graph.remove_edge(n, name)
        for n in list(self.graph.successors(name)):
            self.graph.remove_edge(name, n)
        self.graph.remove_node(name)
        del self._modules[name]

    def add_edge(self, from_, to_):
        self.graph.add_edges_from([(from_, to_)])

    def forward(self, x):
        g = self.graph
        assert 'start' in g, "oops"
        assert 'end' in g, "oops"
        g.nodes['start']['out_shape'] = x.shape
        outputs = {'start': g.nodes['start']['module'](x)}
        queue = list(set(g.adj['start']))
        visited = set('start')
        while queue:
            node_name = queue.pop(0)
            # print(node_name)
            if node_name not in visited:
                if all (pred in outputs for pred in g.predecessors(node_name)):
                    inputs = [outputs[pred] for pred in g.predecessors(node_name)]
                    if len(inputs) == 1:
                        inputs = inputs[0]
                    try:
                        outputs[node_name] = g.nodes[node_name]['module'](inputs)
                    except:
                        import ipdb;ipdb.set_trace()
                    # add attributs
                    # if type(inputs) == list:
                    #     g.nodes[node_name]['in_shapes'] = [x.shape for x in inputs]
                    # else:
                    #     g.nodes[node_name]['in_shape'] = [x.shape for x in inputs]
                    if type(outputs[node_name]) == list:
                        g.nodes[node_name]['out_shape'] = [x.shape for x in outputs[node_name]]
                    else:
                        g.nodes[node_name]['out_shape'] = outputs[node_name].shape

                    visited.add(node_name)
                    queue.extend(set(g.adj[node_name]) - visited)
                else:
                    queue.append(node_name)
        return outputs['end']

#######################################################################################
# Modules that cannot be embedded inside a block

class GlobalAvgPool(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x = torch.mean(x, dim=2)
        return x


class AVG(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        #assert type(x) == list, 'oops'
        if type(x) == list:
            x = sum(x) / len(x)
        return x


# not available with current constraints
class CAT(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if type(x) == list:
            x = torch.cat(x, dim=1)
        return x


class Block(nn.Module):

    def __init__(self, ops, aggr='avg'):
        super().__init__()
        assert type(ops) == list and len(ops) > 1, 'oops'
        self.ops = nn.ModuleList(ops)
        self.aggr = aggr

    def forward(self, x):
        if type(x) == list:
            x = [self.ops[i](x[i]) for i in range(len(self.ops))]
        else:
            x = [self.ops[i](x) for i in range(len(self.ops))]
        if self.aggr == 'avg':
            x = sum(x) / len(x)
        elif self.aggr == 'cat':
            x = torch.cat(x, dim=1)
        else:
            raise ValueError()
        return x


class SepConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # depthwise
        self.dw_conv = nn.Conv2d(in_channels, in_channels,
                                 kernel_size=kernel_size,
                                 stride=stride, padding=padding,
                                 groups=in_channels, bias=False)
        # pointwise
        self.pw_conv = nn.Conv2d(in_channels, out_channels,
                                 kernel_size=1, bias=False)

    def set_stride(self, stride):
        prev_stride = self.dw_conv.stride[0]
        if stride > prev_stride:
            self.dw_conv.stride = (stride, stride)
            self.dw_conv.in_channels = int(self.dw_conv.in_channels / 2)
            self.dw_conv.out_channels = self.dw_conv.in_channels
            self.dw_conv.groups = self.dw_conv.in_channels
            # .data is needed, even in torch v0.4
            self.dw_conv.weight.data = self.dw_conv.weight.data[:self.dw_conv.out_channels,
                                                                :self.dw_conv.in_channels, :, :]
            self.dw_conv.weight.data = self.dw_conv.weight.data.contiguous()
            self.pw_conv.in_channels = int(self.pw_conv.in_channels / 2)
            self.pw_conv.weight.data = self.pw_conv.weight.data[:, :self.pw_conv.in_channels, :, :]
            self.pw_conv.weight.data = self.pw_conv.weight.data.contiguous()
            # TODO: update shape of optimizer as well
        elif stride < prev_stride:
            raise NotImplementedError()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

#######################################################################################
# Modules that can be both (embedded inside a block, or not)

class Identity(nn.Module):

    def __init__(self):
        super().__init__()
        #self.name = 'identity'

    def forward(self, x):
        return x

#######################################################################################
# Modules that can be embedded inside a block

# the stride is used to reduce the spatial dimension by a factor 2
# modules that are compatible for stride 1 and stride 2
stride2_modules = ['cbr_k,3', 'cbr_k,5', 'cbr_k,3_d,2', 'cbr_k,3_d,3', 'cbr_k,3_d,4',
                   'cbr_17_71', 'cbr_71_17', 'sepcbr_k,3', 'sepcbr_k,5', 'sepcbr_k,7']
# modules that are compatible for stride 1 only
stride1_modules = ['identity', 'avg_k,3', 'max_k,3'] + stride2_modules


class CBR(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.name = 'cbr_k,{}'.format(kernel_size)
        if dilation > 1:
            self.name += '_d,{}'.format(dilation)
        #self.stride = stride
        # modules   
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def set_stride(self, stride):
        prev_stride = self.conv.stride[0]
        if stride > prev_stride:
            self.conv.stride = (stride, stride)
            self.conv.in_channels = int(self.conv.in_channels / 2)
            # .data is needed, even in torch v0.4
            self.conv.weight.data = self.conv.weight.data[:, :self.conv.in_channels, :, :]
            self.conv.weight.data = self.conv.weight.data.contiguous()
            # TODO: update shape of optimizer as well
        elif stride < prev_stride:
            raise NotImplementedError()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CBR_17_71(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.name = 'cbr_17_71'
        #self.stride = stride
        # modules
        self.seq = nn.Sequential()
        self.seq.add_module('cbr17', CBR(in_channels, out_channels, (1,7), stride=stride, padding=(0,3)))
        self.seq.add_module('cbr71', CBR(out_channels, out_channels, (7,1), stride=1, padding=(3,0)))

    def set_stride(self, stride):
        self.seq.cbr17.set_stride(stride)

    def forward(self, x):
        x = self.seq(x)
        return x


class CBR_71_17(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.name = 'cbr_71_17'
        #self.stride = stride
        # modules
        self.seq = nn.Sequential()
        self.seq.add_module('cbr71', CBR(in_channels, out_channels, (7,1), stride=stride, padding=(3,0)))
        self.seq.add_module('cbr17', CBR(out_channels, out_channels, (1,7), stride=1, padding=(0,3)))

    def set_stride(self, stride):
        self.seq.cbr71.set_stride(stride)

    def forward(self, x):
        x = self.seq(x)
        return x


class SepCBR(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.name = 'sepcbr_k,{}'.format(kernel_size)
        #self.stride = stride
        # modules
        self.sep_conv = SepConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def set_stride(self, stride):
        self.sep_conv.set_stride(stride)

    def forward(self, x):
        x = self.sep_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# TODO
# compatible stride2 only
# useful for skip-connections
class ReductionCBR(nn.Module):

    def __init__(self, in_channels, kernel_size=1, level=1, padding=0, dilation=1, groups=1):
        super().__init__()
        #assert level >= 0, 'oops'
        self.seq = nn.Sequential()
        out_channels = in_channels
        if level == 0:
            self.seq.add_module(str(level),
                CBR(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=1, groups=1))
        else:
            for l in range(level):
                out_channels = 2 * out_channels
                self.seq.add_module(str(l),
                    CBR(in_channels, out_channels, kernel_size, stride=2, padding=padding, dilation=1, groups=1))

    def forward(self, x):
        x = self.seq(x)
        return x

#######################################################################################
# Utilities

# def custom_layout(g):
#     pos = {}
#     for n in g.nodes:
#         pos[n] = np.array([random.random()*10,random.random()*10])
#     return pos

def dijkstra(g, start='start'):
    dist = {'start': 0}
    queue = list(set(g.adj['start']))
    visited = set('start')
    while queue:
        node_name = queue.pop(0)
        # print(node_name)
        if node_name not in visited:
            if all (pred in dist for pred in g.predecessors(node_name)):
                node_dist = max([dist[pred] for pred in g.predecessors(node_name)]) + 1
                dist[node_name] = node_dist
                visited.add(node_name)
                queue.extend(set(g.adj[node_name]) - visited)
            else:
                queue.append(node_name)
    return dist


def custom_layout(g, start='start', disp_node_dist=2):
    dist = dijkstra(g, start=start)
    nodes_by_dist = collections.defaultdict(list)
    for n, d in dist.items():
        nodes_by_dist[d].append(n)
    pos = {}
    for d in range(len(nodes_by_dist)):
        n_nodes = len(nodes_by_dist[d])
        for i, name in enumerate(sorted(nodes_by_dist[d])):
            x = i * disp_node_dist - n_nodes + 1
            y = -d * disp_node_dist
            y *= 3 # just for beauty
            pos[name] = np.array([x, y])
    return pos


def display(g, figname='test.png'):
    plt.figure(figsize=(10,20), dpi=300)
    plt.subplot(111)
    #pos = nx.kamada_kawai_layout(g, scale=2)
    pos = custom_layout(g)
    labels = {}
    for name, node in g.nodes.items():
        if type(node['module']) is Block:
            op0 = node['module'].ops[0]
            op1 = node['module'].ops[1]
            op0_repr = op0.name if hasattr(op0, 'name') else type(op0).__name__
            op1_repr = op1.name if hasattr(op1, 'name') else type(op1).__name__
            module_repr = 'Block({}, {})'.format(op0_repr, op1_repr)
        else:
            module_repr = type(node['module']).__name__
        shape_repr = str(list(node['out_shape'][1:]))
        labels[name] = '{}\n{}\n{}'.format(name, module_repr, shape_repr)
    nx.draw(g, pos=pos,
        #with_labels=True,
        node_size=600,
        node_shape='o',#'o',
        labels=labels,
        font_size=3,
        font_weight='bold')
    plt.savefig(figname)


def bfs(graph, start):
    path = []
    visited, queue = set(), [start]
    while queue:
        node_name = queue.pop(0)
        if node_name not in visited:
            visited.add(node_name)
            path.append(node_name)
            queue.extend(set(graph.adj[node_name]) - visited)
    return path


def add_cell(net, name, stride=1, in_channels=128, out_channels=256, from_='cell_0,end'):
    print('add {}'.format(name))
    net.add_edge(from_, name+',start')
    net.add_node(name+',start', Identity())
    # log the stride as an attribute of start node
    net.graph.nodes[name+',start']['stride'] = stride

    net.add_edge(name+',start', name+',block_0')
    net.add_node(name+',block_0', Block([
        CBR(in_channels, out_channels, 3, stride=stride, padding=1),
        CBR(in_channels, out_channels, 5, stride=stride, padding=2),
        #nn.AvgPool2d(3, stride=2, padding=1)
    ]))

    # net.add_edge(name+',start', name+',block_1')
    # net.add_node(name+',block_1', Block([
    #     CBR(in_channels, out_channels, 3, stride=stride, padding=1),
    #     CBR(in_channels, out_channels, 5, stride=stride, padding=2),
    #     #nn.AvgPool2d(3, stride=2, padding=1)
    # ]))

    # # BEWARE OF THE ORDER
    # net.add_edge(name+',block_0', name+',block_2')
    # net.add_edge(name+',block_1', name+',block_2')
    # net.add_node(name+',block_2', Block([
    #     CBR(out_channels, out_channels, 3, stride=1, padding=1),
    #     CBR(out_channels, out_channels, 3, stride=1, padding=1)
    #     #nn.AvgPool2d(3, stride=1, padding=1)
    # ]))
    
    net.add_edge(name+',block_0', name+',end')
    net.add_node(name+',end', AVG())


def get_blocks_in_cell(g, cname='cell_0'):
    pattern = '{},(block_[0-9]+)'.format(cname)
    bnames = [re.findall(pattern, n)[0] for n in g.graph.nodes \
              if len(re.findall(pattern, n)) > 0]
    return bnames


def get_cells_in_graph(g):
    pattern = 'cell_[0-9]+'
    cnames = [re.findall(pattern, n)[0] for n in g.graph.nodes \
              if len(re.findall(pattern, n)) > 0]
    cnames = list(set(cnames)) # unique
    return cnames


# def pick_block(g, cname='cell_0'):
#     bnames = get_blocks_in_cell(cname)
#     bid = int(random.random()*len(bnames))
#     return bnames[bid]


# # TODO: testing
# def get_in_channels(g, name):
#     if type(g.nodes[name]['module']) == Block:
#         for op in g.nodes[name]['module'].ops:
#             if type(op) == CBR:
#                 return op.conv.in_channels
#         for op in g.nodes[name]['module'].ops:
#             if type(op) in [Identity, CAT, AVG]:
#                 for n in g.predecessors(name):
#                     out_channels = get_out_channels(g, n)
#                     if out_channels is not None:
#                         return out_channels
#     elif type(g.nodes[name]['module']) in [Identity, CAT, AVG]:
#         for n in g.predecessors(name):
#             out_channels = get_out_channels(g, n)
#             if out_channels is not None:
#                 return out_channels
#     return None

# def get_in_channels(g, name):
#     return g.nodes[name]['in_shape'][1]

# # TODO: testing
# def get_out_channels(g, name):
#     if type(g.nodes[name]['module']) == Block:
#         for op in g.nodes[name]['module'].ops:
#             if type(op) == CBR:
#                 return op.conv.out_channels
#         for op in g.nodes[name]['module'].ops:
#             if type(op) in [Identity, CAT, AVG]:
#                 for n in g.successors(name):
#                     in_channels = get_in_channels(g, n)
#                     if in_channels is not None:
#                         return in_channels
#     elif type(g.nodes[name]['module']) in [Identity, CAT, AVG]:
#         for n in g.successors(name):
#             in_channels = get_in_channels(g, n)
#             if in_channels is not None:
#                 return in_channels
#     return None

def get_out_channels(g, name):
    return g.graph.nodes[name]['out_shape'][1]


def pick_module_name(stride=1):
    if stride == 1:
        name = random.choice(stride1_modules)
    elif stride == 2:
        name = random.choice(stride2_modules)
    else:
        raise ValueError()
    return name


def pick_module_pair():
    ''' pick one module M1 stride 1 compatible
        if it is stride 2 compatible as well, return (M1, M1)
        else pick M2 from stride 2 compatible, return (M1, M2)
    '''
    n = {}
    n[1] = pick_module_name(stride=1)
    if n[1] in stride2_modules:
        n[2] = n[1]
    else:
        n[2] = pick_module_name(stride=2)
    return n


def get_module(name, in_channels=-1, out_channels=-1, stride=1):
    # TODO: 1x1 conv, cbr 13_31 etc
    # modules that are compatible stride 1 and 2
    if name == 'cbr_k,3':
        m = CBR(in_channels, out_channels, 3, stride=stride, padding=1)
    elif name == 'cbr_k,5':
        m = CBR(in_channels, out_channels, 5, stride=stride, padding=2)
    elif name == 'cbr_k,3_d,2':
        m = CBR(in_channels, out_channels, 3, stride=stride, padding=2, dilation=2)
    elif name == 'cbr_k,3_d,3':
        m = CBR(in_channels, out_channels, 3, stride=stride, padding=3, dilation=3)
    elif name == 'cbr_k,3_d,4':
        m = CBR(in_channels, out_channels, 3, stride=stride, padding=4, dilation=4)
    elif name == 'cbr_17_71':
        m = CBR_17_71(in_channels, out_channels, stride=stride)
    elif name == 'cbr_71_17':
        m = CBR_71_17(in_channels, out_channels, stride=stride)
    elif name == 'sepcbr_k,3':
        m = SepCBR(in_channels, out_channels, 3, stride=stride, padding=1)
    elif name == 'sepcbr_k,5':
        m = SepCBR(in_channels, out_channels, 5, stride=stride, padding=2)
    elif name == 'sepcbr_k,7':
        m = SepCBR(in_channels, out_channels, 7, stride=stride, padding=3)
    # compatible stride 1 only
    elif name == 'identity':
        assert in_channels == out_channels
        m = Identity()
    elif name == 'avg_k,3':
        assert in_channels == out_channels
        m = nn.AvgPool2d(3, stride=stride, padding=1)
    elif name == 'max_k,3':
        assert in_channels == out_channels
        m = nn.MaxPool2d(3, stride=stride, padding=1)
    else:
        raise ValueError()
    return m


def first_missing_block(current_bnames, max_blocks):
    for i in range(max_blocks+1): # beware, TODO: raise ValueError?
        bname = 'block_{}'.format(i)
        if bname not in current_bnames:
            break
    return bname


def get_prev_cell(cname):
    return 'cell_{}'.format(int(cname[-1]) - 1)


def add_block_in_cells(net):
    MAX_BLOCKS = 10

    current_bnames = get_blocks_in_cell(net, cname='cell_0')
    if len(current_bnames) >= MAX_BLOCKS:
        raise ValueError()

    bname_new = first_missing_block(current_bnames, MAX_BLOCKS)
    # pick two inputs
    bnames_in = ['__prev_cell__'] # TODO: skip-connections
    bnames_in += ['start']
    bnames_in += current_bnames
    bname_in0 = random.choice(bnames_in)
    bname_in1 = random.choice(bnames_in)

    module_names0 = pick_module_pair()
    module_names1 = pick_module_pair()

    for cname in get_cells_in_graph(net):
        name_sta = '{},start'.format(cname)
        name_new = '{},{}'.format(cname, bname_new)
        name_end = '{},end'.format(cname)
        cell_stride = net.graph.nodes[name_sta].get('stride', 1)

        # get module0
        if bname_in0 == '__prev_cell__' and cname == 'cell_0':
            name_in0 = '{},{}'.format(cname, 'start')
        elif bname_in0 == '__prev_cell__' and cname != 'cell_0':
            name_in0 = '{},{}'.format(get_prev_cell(cname), 'start')
        else:
            name_in0 = '{},{}'.format(cname, bname_in0)

        in_channels0 = get_out_channels(net, name_in0)
        stride0 = net.graph.nodes[name_in0].get('stride', 1)
        if bname_in0 == '__prev_cell__' and cname != 'cell_0':
            level = (stride0-1) + (cell_stride-1)
            module0 = ReductionCBR(in_channels0, level=level)
        else:
            module_name0 = module_names0[stride0]
            out_channels0 = in_channels0 * stride0
            module0 = get_module(module_name0, in_channels0, out_channels0, stride=stride0)

        # get module1
        if bname_in1 == '__prev_cell__' and cname == 'cell_0':
            name_in1 = '{},{}'.format(cname, 'start')
        elif bname_in1 == '__prev_cell__' and cname != 'cell_0':
            name_in1 = '{},{}'.format(get_prev_cell(cname), 'start')
        else:
            name_in1 = '{},{}'.format(cname, bname_in1)
        
        in_channels1 = get_out_channels(net, name_in1)
        stride1 = net.graph.nodes[name_in1].get('stride', 1)
        if bname_in1 == '__prev_cell__' and cname != 'cell_0':
            level = (stride1-1) + (cell_stride-1)
            module1 = ReductionCBR(in_channels1, level=level)
        else:
            module_name1 = module_names1[stride1]
            out_channels1 = in_channels1 * stride1
            module1 = get_module(module_name1, in_channels1, out_channels1, stride=stride1)

        net.add_edge(name_in0, name_new)
        if name_in0 != name_in1: # would replace the last edge, because same keys
            net.add_edge(name_in1, name_new)
        net.add_edge(name_new, name_end)
        net.add_node(name_new, Block([module0, module1]))
    return bname_new


#def convert_op_to_stride2()

def remove_block_in_cells(net):
    MIN_BLOCKS = 1
    current_bnames = get_blocks_in_cell(net, cname='cell_0')
    if len(current_bnames) <= MIN_BLOCKS:
        raise ValueError()

    bname_del = random.choice(current_bnames)

    for cname in get_cells_in_graph(net):
        name_sta = '{},start'.format(cname)
        name_del = '{},{}'.format(cname, bname_del)
        name_end = '{},end'.format(cname)

        net.remove_node(name_del)

        # verify block viability
        for bname_via in get_blocks_in_cell(net, cname=cname):
            # if needed, add an edge from the start node of the current cell
            name_via = '{}.{}'.format(cname, bname_via)           
            n_preds = len(list(net.graph.predecessors(name_via)))

            if n_preds == 0:
                # no predecessors, then add an edge from start
                # and mutates left and right operations of block if needed
                net.add_edge(name_sta, name_via)
                mutate_op_ids = [0,1]

            elif n_preds == 1 and name_sta not in net.graph._pred[name_via]:
                # 1 predecessor which is not the start node, then add an edge from start
                # and mutates the operation pointed by the new edge
                net.add_edge(name_sta, name_via)
                idx = net.graph._pred[name_via].order[name_sta]
                mutate_op_ids = [idx]

            elif n_preds == 1 and name_sta in net.graph._pred[name_via]:
                # 1 predecessor which is the start node, then do not add edge from start
                # (because only one edge with the same key can exist in the graph)
                # and mutates the other operation (that would have been pointed by the new edge)
                idx = net.graph._pred[name_via].order[name_sta]
                idx = (idx + 1) % 2
                mutate_op_ids = [idx]

            elif n_preds == 2:
                # adding predecessors is not needed
                continue
            else:
                raise ValueError()

            cell_stride = net.graph.nodes[name_sta].get('stride', 1)
            if cell_stride == 2:
                # if needed, mutate the operations from stride 1 to 2
                for i in mutate_op_ids:
                    op = net.graph.nodes[name_via]['module'].ops[i]
                    if hasattr(op, 'set_stride'):
                        op.set_stride(2)
                    else:
                        # if current operation is not stride 2 compatible
                        # replace it by a new one which is compatible
                        module_name = random.choice(stride2_modules)
                        in_channels = get_out_channels(net, name_sta)
                        out_channels = in_channels * cell_stride
                        op = get_module(module_name, in_channels, out_channels, stride=cell_stride)
                        net.graph.nodes[name_via]['module'].ops[i] = op
                        #import ipdb;ipdb.set_trace()

            # if needed, add edges to end node
            n_succs = len(list(net.graph.successors(name_via)))
            for i in range(1, n_succs, -1):
                net.add_edge(name_via, name_end)

    return bname_del


def modify_block(g, bname='cell_0.block_0', mutation=None):
    pass



# def test_get_in_channels()
#     g.add_node('start', module=Identity())
#     g.add_edge('start', 'cbr')
#     g.add_node('cbr', module=CBR(3, 128, 3, stride=2, padding=1))
#     add_cell(g, 'cell_0', stride=1, in_channels=128, out_channels=256, from_='cbr')
#     add_cell(g, 'cell_1', stride=1, in_channels=256, out_channels=256, from_='cell_0.end')
#     add_cell(g, 'cell_2', stride=1, in_channels=256, out_channels=256, from_='cell_1.end')
#     add_cell(g, 'cell_3', stride=1, in_channels=256, out_channels=256, from_='cell_2.end')
#     g.add_edge('cell_3.end', 'pool')
#     g.add_node('pool', module=GlobalAvgPool())
#     g.add_edge('pool', 'end')
#     g.add_node('end', module=Identity())
#     get_in_channels(g, 'end')


# if __name__ == '__main__':
#     conv = nn.Conv2d(10, 20, 3, stride=1, padding=2, dilation=2)
#     conv = nn.Conv2d(10, 20, 3, stride=1, padding=3, dilation=3)
#     conv = nn.Conv2d(10, 20, 3, stride=1, padding=4, dilation=4)
#     conv = nn.Conv2d(10, 20, 3, stride=2, padding=2, dilation=2)
#     x = torch.randn(1,10,50,50)
#     o = conv(x)
#     print(o.shape)

    

def TestNet():
    print('# create the graph')
    net = GraphModule()

    net.add_node('start', Identity())
    net.add_edge('start', 'cbr')
    net.add_node('cbr', CBR(3, 10, 3, stride=2, padding=1))
    add_cell(net, 'cell_0', stride=1, in_channels=10, out_channels=10, from_='cbr')
    add_cell(net, 'cell_1', stride=2, in_channels=10, out_channels=20, from_='cell_0,end')
    add_cell(net, 'cell_2', stride=1, in_channels=20, out_channels=20, from_='cell_1,end')
    add_cell(net, 'cell_3', stride=1, in_channels=20, out_channels=20, from_='cell_2,end')
    # add_cell(g, 'cell_4', stride=1, in_channels=256, out_channels=512, from_='cell_3.end')
    # add_cell(g, 'cell_5', stride=1, in_channels=512, out_channels=512, from_='cell_4.end')
    # add_cell(g, 'cell_6', stride=1, in_channels=512, out_channels=512, from_='cell_5.end')
    # add_cell(g, 'cell_7', stride=1, in_channels=512, out_channels=512, from_='cell_6.end')
    net.add_edge('cell_3.end', 'pool')
    net.add_node('pool', GlobalAvgPool())
    net.add_edge('pool', 'end')
    net.add_node('end', Identity())
    return net



if __name__ == '__main__':
    seed = int(random.random()*1000)
    print('seed', seed)
    random.seed(seed)
    input = torch.randn(1,4,416,640).cuda()
    #input = torch.randn(1,3,200,200).cuda()
    net = GraphInceptionV1()
    #import ipdb;ipdb.set_trace()
    print('Number of params', count_parameters(net))

    net.cuda()
    output = net(input)
    if type(output) == list:
        for x in output:
            print(x.shape)
    else:
        print('Output shape: {}'.format(list(output.shape)))
    n_mutations = 0
    display(net.graph, figname='test_0.png')
    print('Num nodes: {}'.format(len(net.graph.nodes)))

    state = net.state_dict()

    del net

    net = GraphModule()
    net.load_state_dict(state)

    net.cuda()
    output = net(input)
    if type(output) == list:
        for x in output:
            print(x.shape)
    else:
        print('Output shape: {}'.format(list(output.shape)))

    # for i in range(7):
    #     n_mutations += 1
    #     print('\nMutation #{} -> ADD {}'.format(
    #         n_mutations, add_block_in_cells(net)))
    #     net.cuda()
    #     output = net(input)
    #     print('Output shape: {}'.format(list(output.shape)))
    #     display(net.graph, figname='test_{}.png'.format(n_mutations))
    #     print('Num nodes: {}'.format(len(net.graph.nodes)))
 
    # for i in range(9):
    #     n_mutations += 1
    #     print('\nMutation #{} -> DEL {}'.format(
    #         n_mutations, remove_block_in_cells(net)))
    #     net.cuda()
    #     output = net(input)
    #     print('Output shape: {}'.format(list(output.shape)))
    #     display(net.graph, figname='test_{}.png'.format(n_mutations))
    #     print('Num nodes: {}'.format(len(net.graph.nodes)))




    

