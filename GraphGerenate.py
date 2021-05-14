import numpy
import logging
from NiuTensor2onnx import read_data, To_onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[NiuTensor2TREE]")

# Elementwise operation
kElemWise = 0
# Broadcasting operator, can always map output axis to the input in order.
# for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
# Note that the axis need to be in order so transpose is not a bcast operator.
kBroadcast = 1
# Injective operator, can always injectively map output axis to a single input axis.
# All injective operator can still be safely fused to injective and reduction.
kInjective = 2
# Communicative reduction operator.
kCommReduce = 3
# Complex operation, can still fuse elemwise operations into its output.
# but cannot chain another complex op
kOutEWiseFusable = 4
# The pattern for tuple nodes. Can fuse into subsequent injective ops,
# but treated specially
kTuple = 7
# Opaque operation, cannot fuse anything.
kOpaque = 8

# OpPatternKind edge_pattern = op_pattern;
# if (edge_pattern == kBroadcast && arg_type != nullptr && rtype != nullptr &&
#     attr_equal_(rtype->shape, arg_type->shape)) {
#     edge_pattern = kElemWise;
# }


NiuTensor_OPPATTERN = {'L_CROSSENTROPY': kBroadcast,
                       'M_GATHER': kBroadcast,
                       'F_RECTIFY': kOpaque,
                       'M_MULTIPLY_I': kBroadcast,
                       'M_MULTIPLYDIM': kBroadcast,
                       'M_SUMDIM': kBroadcast,
                       'M_MATRIXMUL': kBroadcast,
                       'R_REDUCEVARIANCE': kCommReduce,
                       'M_MULTIPLY': kBroadcast,
                       'S_SPLIT': kInjective,
                       'S_UNSQUEEZE': kInjective,
                       'F_SOFTMAX': kBroadcast,
                       'M_OPERATION': kBroadcast,
                       'M_SUB': kBroadcast,
                       'R_REDUCEMEAN': kCommReduce,
                       'M_MATRIXMULBATCHED': kOpaque,
                       'M_POWER': kBroadcast,
                       'M_SCALE': kBroadcast,
                       'M_SUM': kBroadcast,
                       'S_MERGE': kCommReduce,
                       'M_DIV': kBroadcast
                       }


class GraphNode:
    def __init__(self):
        # 节点名称
        self.name = None
        # 节点的边
        self.outputs = []
        self.index = 0
        # weak reference to the corresponding edge
        self.ref = None
        # 代表该节点是否是根节点，extern_ref = 1代表根节点
        self.extern_ref = 0
        self.pattern = kOpaque


class LinkNode:
    def __init__(self):
        self.value = None
        self.pattern = 0
        self.next = None


class Graph:
    def __init__(self):
        # 根据node的index来寻找对应的图节点
        self.edge_node_dict = {}
        # DAG图的倒序dfs序列
        self.post_dfs_order = []
        self.visited_list = []
        self.added_dict = {}
        self.root_flag = 1

    def FindNode(self, node_index):
        for init in constant_nodes:
            if node_index == init['index']:
                # print(init)
                return init, "var"
        for item in nodes:
            if node_index == item['index']:
                # print(item)
                return item, "node"
        logger.info("cannot find node {0}".format(node_index))
        # exit(1)

    def Update(self, node, parent, pattern):
        '''
        通过node数组，创建图节点和每个图节点对应的边，建立DAG图
        '''
        # 如果这个节点已经有了对应的图节点，直接通过edge_node_dict字典找到它对应的图节点
        if node['index'] in self.edge_node_dict.keys():
            current = self.edge_node_dict[node['index']]
            # print("[update] {0}".format(node.name))
        else:
            # 但前节点还没有对应的图节点，创建一个新的图节点
            current = GraphNode()
        if node in nodes:
            if parent is not None:
                link = LinkNode()
                if parent['index'] not in self.edge_node_dict.keys():
                    logger.error("cannot find node {0} in edge dict, prob this is the last node".format(parent.name))
                    exit(1)
                parent = self.edge_node_dict[parent['index']]
                link.value = parent
                link.pattern = pattern
                current.name = node['index']
                current.outputs.append(link)
            else:
                current.name = node['index']
                current.extern_ref = 1
        return current

    def AddNode(self, node, node_pattern):
        if node['index'] not in self.edge_node_dict.keys():
            logger.error("cannot find node {0} in edge dict, prob this is the last node".format(node.name))
            exit(1)
        current = self.edge_node_dict[node['index']]
        current.index = len(self.post_dfs_order)
        current.ref = node
        current.pattern = node_pattern
        # print(current.outputs[0].value.pattern)
        logger.info("[add node] {0} {1} ".format(current.index, node['index']))
        # the node is not added to dfs_order
        if node['index'] not in self.added_dict.keys():
            # logger.info("======================")
            # logger.info("[add node] {0}".format(node.name))
            # logger.info("======================")
            self.post_dfs_order.append(current)
            self.added_dict[node['index']] = current.index
        # else:
        #     index = self.added_dict[node['index']]
        #     self.post_dfs_order[index] = current

    def VisitExpr(self, node):
        '''
        遍历node数组，建立DAG图
        param {
            node: 根节点
        }
        '''
        print(node)
        # if the node had been visited, return back
        if node is None or node in self.visited_list:
            return
            # create the root node and add to dict
        if self.root_flag:
            edge_root_node = self.Update(node, None, kOpaque)
            self.edge_node_dict[node['index']] = edge_root_node
            self.root_flag = 0
        if int(node['inputnum']) != 0:
            print(kBroadcast)
            print(node['operator'], NiuTensor_OPPATTERN[node['operator']])
            op_pattern = NiuTensor_OPPATTERN[node['operator']]
            print(op_pattern)
        else:
            op_pattern = kOpaque
        # print(node['inputnode'])
        for input_s in node['inputnode']:
            edge_pattern = op_pattern
            # here assum all output shape of bn and add node is keep same
            # if edge_pattern ==  kBroadcast:
            #     edge_pattern =  kElemWise
            # print(input_s)
            # if input_s == "":
            #     break
            input_node, node_type = self.FindNode(input_s)
            # print(node_type)
            if node_type == "node":
                # if input_node not in self.visited_list:
                edge_node = self.Update(input_node, node, edge_pattern)
                self.edge_node_dict[input_node['index']] = edge_node
                self.VisitExpr(input_node)
                self.visited_list.append(input_node)
                # else:
                #     edge_leaf_root_node = self.Update(input_node, None, op_pattern)
                #     self.edge_node_dict[input_node.name] = edge_leaf_root_node
            elif node_type == "var":
                self.visited_list.append(input_node)
        self.AddNode(node, op_pattern)
        return

# Group用来表示哪些算子融合在一起
class Group:
    def __init__(self):
        self.parent = None
        self.pattern = 0    #当前group的类型，也就是融合后算子的类型
        self.root_ref = None    #当前group的根节点
        self.master_ref = None
        self.name = None
        self.num_nodes = 1

    # 路径压缩，并返回当前节点所在group的根节点
    def FindRoot(self):
        if self.parent is None:
            return self
        else:
            root = self
            # 当前节点还有父节点，说明它不是group的根节点
            while root.parent is not None:
                root = root.parent
            # 把这一个group中所有节点的parent设置成这个group的根节点
            while self is not root:
                parent = self.parent
                self.parent = root
                self = parent
        return root


class DominatorTree:
    '''
    支配树包含两个部分
    第一个部分是group数组，表示每一个算子的分组情况
    第二个部分就是tree_nodes数组，用来表示每一个算子的
    '''
    def __init__(self):
        super().__init__()
        self.groups = []
        self.tree_nodes = []

    class TreeNode:
        def __init__(self):
            self.name = None
            self.parent = None
            self.depth = 0
            self.pattern = kOpaque
            self.index = 0
            # 对应的图节点
            self.gnode = None

    def InitGroups(self, graph):
        size = len(graph.post_dfs_order)
        # 为每个graph node建立一个group, 表示尚未融合的状态
        for index in range(size):
            graph_node = graph.post_dfs_order[index]
            group_node = Group()
            group_node.pattern = graph_node.pattern
            group_node.root_ref = graph_node.ref
            group_node.name = graph_node.name
            if group_node.pattern == kOutEWiseFusable:
                group_node.master_ref = graph_node.ref
            self.groups.append(group_node)
            # logger.info(group_node, graph_node.index)

    # return the max pattern
    def CombinePattern(self, lhs, rhs):
        # print(lhs, rhs)
        if lhs > rhs:
            return lhs
        return rhs

    def LeastCommonAncestorMulEdges(self, lhs, rhs, edge_pattern):
        while lhs != rhs:
            if lhs is None:
                return None
            if rhs is None:
                return None
            if lhs.depth < rhs.depth:
                edge_pattern = self.CombinePattern(edge_pattern, rhs.pattern)
                rhs = rhs.parent
            elif rhs.depth < lhs.depth:
                edge_pattern = self.CombinePattern(edge_pattern, lhs.pattern)
                lhs = lhs.parent
            else:
                edge_pattern = self.CombinePattern(edge_pattern, lhs.pattern)
                edge_pattern = self.CombinePattern(edge_pattern, rhs.pattern)
                lhs = lhs.parent
                rhs = rhs.parent
        return lhs

    def LeastCommonAncestor(self, edges, edge_pattern, index):
        if len(edges) <= index:
            return None
        # 该节点对应的第一条边
        link_head = edges[index]

        # 通过graph node的Index找到对应的tree node
        def get_node(father_node):
            oindex = father_node.index
            return self.tree_nodes[oindex]

        parent = get_node(link_head.value)

        edge_pattern = self.CombinePattern(edge_pattern, link_head.value.pattern)
        index = index + 1
        # 多边的情况，处理剩余的边
        for i in range(index, len(edges)):
            link = edges[i]
            parent = self.LeastCommonAncestorMulEdges(parent, get_node(link.value), edge_pattern)
            edge_pattern = self.CombinePattern(edge_pattern, link.value.pattern)
        return parent, edge_pattern

    def GetNode(self, graph_node, graph):
        tree_node = self.TreeNode()
        tree_node.gnode = graph_node
        # 该节点没有parent节点（对应的graph上没有出度）
        if graph_node.extern_ref == 1:
            tree_node.name = graph_node.name
            tree_node.depth = 1
            tree_node.parent = None
            tree_node.pattern = kOpaque
            # tree_node.parent_gnode = graph_node
        else:
            # find the LCAs of all outputs.
            pattern = kElemWise
            tree_node.name = graph_node.name
            # 该graph node对应的所有parent node中最大的pattern
            parent, pattern = self.LeastCommonAncestor(graph_node.outputs, pattern, 0)
            tree_node.depth = parent.depth + 1 if parent else 1
            tree_node.parent = parent
            tree_node.pattern = pattern
            # parent_gnode = None
            # for node in graph:
            #     if node.name == parent.name:
            #         parent_gnode = node
            # assert parent_gnode is not None
            # tree_node.parent_gnode = parent_gnode
            logger.info(
                "[dom node] {0}\t\t{1}\t\t{2}".format(tree_node.depth, graph_node.name, tree_node.parent.gnode.name))
        return tree_node

    def PostDom(self, graph):
        size = len(graph.post_dfs_order)
        self.tree_nodes = [None] * size
        # self.tree_nodes[0] = self.GetNode(graph.post_dfs_order[0])
        # 逆序遍历dfs数组
        for i in range(size, 0, -1):
            # print(graph.post_dfs_order[i - 1].name)
            self.tree_nodes[i - 1] = self.GetNode(graph.post_dfs_order[i - 1], graph.post_dfs_order)

    def DominatorPartition(self, graph):
        self.InitGroups(graph)
        # print(self.groups)
        self.PostDom(graph)


class FuseOps:
    def __init__(self):
        self.fuse = None
        self.visited = []

    def CheckPath_(self, src, sink, fcond, tree):
        # print(type(src), type(sink))
        # print(src.name)
        if src in self.visited:
            return True
        self.visited.append(src)
        gnode = tree.groups[src.index]
        assert gnode is not None
        gnode = gnode.FindRoot()
        if not fcond(gnode.pattern, src == sink):
            return False
        if src == sink:
            return True
        for link in src.outputs:
            if not self.CheckPath_(link.value, sink, fcond, tree):
                return False
        return True

    def CheckPath(self, src, sink, fcond, tree):
        # print(src.name, src.extern_ref)
        assert src.extern_ref == 0, "root node, error"
        self.visited = []
        assert src != sink
        for link in src.outputs:
            if not self.CheckPath_(link.value, sink, fcond, tree):
                return False
        return True

    def MergeFromTo(self, child, parent):
        child = child.FindRoot()
        parent = parent.FindRoot()
        # logger.info(child.name, parent.name)
        if child == parent:
            return
        parent.num_nodes += child.num_nodes
        child.parent = parent
        # print(parent.master_ref)
        if child.master_ref is not None:
            # logger.error("[Merge] ", child.name, parent.name)
            assert parent.master_ref is None
            parent.master_ref = child.master_ref
            parent.pattern = child.pattern
        # else:
        #     assert parent.master_ref is not None
        #     child.master_ref = parent.master_ref
        #     child.pattern = parent.pattern

    def CommitFuse_(self, src, sink, target, tree):
        if src == sink:
            return
        if src in self.visited:
            return
        self.visited.append(src)
        gnode = tree.groups[src.index]
        assert gnode is not None
        self.MergeFromTo(gnode, target)
        for link in src.outputs:
            self.CommitFuse_(link.value, sink, target, tree)

    def CommitFuse(self, src, sink, tree):
        target = tree.groups[sink.index]
        logger.info("[Merge] {0} + {1} -> {2}".format(src.name, sink.name, target.name))
        self.visited = []
        assert src != sink
        self.CommitFuse_(src, sink, target, tree)

    def RunFuse(self, graph, tree, phase):
        # insgesamt 3 phase to fuse ops, that means 3 methods
        def fcond0(kind, issink):
            # conv + elemwise -> fused-conv-elemwise
            return kind <= kBroadcast

        def fcond1(kind, issink):
            if not issink:
                return kind <= kInjective
            else:
                return (
                        kind <= kBroadcast or kind == kCommReduce or kind == kInjective or kind == kOutEWiseFusable)

        def fcond2(kind, issink):
            return kind <= kInjective

        # print(len(tree.groups))
        for i in range(0, len(tree.groups)):
            graph_node = graph.post_dfs_order[i]
            dom_node = tree.tree_nodes[i] # 第 i 个节点的立即支配点
            group_node = tree.groups[i] #第 i 个节点所属的group
            # 该节点类型为opaque，不可融合，跳过
            if group_node.pattern == kOpaque:
                continue
            # print(group_node.pattern)
            # 该节点的支配点没有父节点，无法进行融合，跳过
            if dom_node.parent is None:
                continue
            # if CountFusedNodesWithNewChild(graph_node, dom_node.parent_gnode) > max_fuse_depth:
            #     continue
            # 当前节点已经与支配点属于同一个group,不用进行融合
            dom_node_parent_gnode_index = dom_node.parent.gnode.index
            if tree.groups[
                dom_node_parent_gnode_index] is not None and group_node.FindRoot() == tree.groups[
                dom_node_parent_gnode_index].FindRoot():
                continue
            #第一次循环，执行第一条融合规则
            if group_node.pattern == kOutEWiseFusable:
                if phase != 0:
                    continue
                if dom_node.parent is not None and dom_node.pattern == kElemWise:
                    logger.info("[fuse node] {0} {1}".format(group_node.name, dom_node.parent.name))
                    if self.CheckPath(graph_node, dom_node.parent.gnode, fcond0, tree):
                        self.CommitFuse(graph_node, dom_node.parent.gnode, tree)
            elif group_node.pattern <= kBroadcast:
                if dom_node.parent is not None and (
                        dom_node.pattern <= kInjective or dom_node.pattern == kCommReduce):
                    logger.info("[fuse node] {0} {1}".format(group_node.name, dom_node.parent.name))
                    if self.CheckPath(graph_node, dom_node.parent.gnode, fcond1, tree):
                        self.CommitFuse(graph_node, dom_node.parent.gnode, tree)
            elif group_node.pattern == kInjective:
                if phase != 1:
                    continue
                logger.info("[fuse node] {0} {1}".format(group_node.name, dom_node.parent.name))
                if self.CheckPath(graph_node, dom_node.parent.gnode, fcond2, tree):
                    self.CommitFuse(graph_node, dom_node.parent.gnode, tree)


def Init(NiuTensor_nodes, Constant_nodes):
    nodes = []
    constant_nodes = []
    for item in NiuTensor_nodes:
        node = {}
        node['index'] = item.index
        node['inputnum'] = item.incomenum
        node['inputnode'] = item.income
        node['outputnum'] = item.outgonum
        node['outputnode'] = item.outgo
        node['operator'] = item.operator
        nodes.append(node)
    for item in Constant_nodes:
        node = {}
        node['index'] = item.index
        node['inputnum'] = item.incomenum
        node['inputnode'] = item.income
        node['outputnum'] = item.outgonum
        node['outputnode'] = item.outgo
        # node['operator'] = item.operator
        constant_nodes.append(node)
    # opset = set(op)
    print(len(nodes), len(constant_nodes))
    # print(constant_nodes)
    return nodes, constant_nodes


def SaveFusedGraph(open_file_name, save_file_name):
    def find_node(node_index):
        for i in range(0, len(nodes)):
            if nodes[i]['index'] == node_index:
                return i, nodes[i], 'node'
        for i in range(0, len(constant_nodes)):
            if constant_nodes[i]['index'] == node_index:
                return i, constant_nodes[i], 'var'
        return -1, None, ""

    def count_group(node_index_list):
        exist_groups_name = []
        # print(node_index_list)
        for node_index in node_index_list:
            id, node, type = find_node(node_index)
            if node is None:
                logging.error("Can't find the node index {0}".format(node_index))
                exit(1)
            if type == "var":
                # print(node['index'])
                exist_groups_name.append(node['index'])
            else:
                root_node_name = post_dom_tree.groups[num - id - 1].FindRoot().name
                # print(post_dom_tree.groups[num - id - 1].name, node_index, root_node_name)
                if root_node_name in exist_groups_name:
                    continue
                else:
                    exist_groups_name.append(root_node_name)
        return exist_groups_name

    def unite_group(root_node_name):
        group = []
        for i in range(num - 1, -1, -1):
            if root_node_name == post_dom_tree.groups[i].FindRoot().name:
                group.append(post_dom_tree.groups[i].name)
        return group

    num = len(nodes)
    print(num)
    node_dict = {}
    for i in range(num - 1, -1, -1):
        print(i)
        if nodes[num - i - 1]['index'] != post_dom_tree.groups[i].FindRoot().name:
            continue
        group = unite_group(nodes[num - i - 1]['index'])
        print(group)
        input_group = []
        output_group = []
        for j in group:
            id, node, type = find_node(j)
            input_group += nodes[id]['inputnode']
            output_group += nodes[id]['outputnode']
        print(input_group, output_group)
        # print(nodes[num - i - 1]['inputnode'])

        groups = count_group(input_group)
        if nodes[num - i - 1]['index'] in groups:
            groups.remove(nodes[num - i - 1]['index'])
        # print(groups)
        str = "node {0} - income[{1}, {2}]: ".format(nodes[num - i - 1]['index'], len(groups),
                                                     'Fused-' + nodes[num - 1 - i]['operator'])
        if len(groups) == 0:
            str += 'null '
        else:
            for id in groups:
                str += id + " "
        groups = count_group(output_group)
        if nodes[num - i - 1]['index'] in groups:
            groups.remove(nodes[num - i - 1]['index'])
        str += ", outgo[{0}]:".format(len(groups))
        if len(groups) == 0:
            str += ' null'
        else:
            for id in groups:
                str += " " + id
        print(str)
        node_dict[nodes[num - i - 1]['index']] = str

    cons_num = len(constant_nodes)
    for i in range(cons_num - 1, -1, -1):
        str = "node {0} - income[0]: null ".format(constant_nodes[i]['index'])
        groups = count_group(constant_nodes[i]['outputnode'])
        str += ", outgo[{0}]:".format(len(groups))
        if len(groups) == 0:
            str += ' null'
        else:
            for id in groups:
                str += " " + id
        print(str)
        node_dict[constant_nodes[i]['index']] = str

    file = open(open_file_name, "r")
    file_write = open(save_file_name, "w")
    text = file.read()
    text = text.split('\n')
    text = list(filter(None, text))
    for i in range(0, len(text), 2):
        info = text[i]
        value = text[i + 1]
        node_index = info.split(' ')[1]
        id, node, type = find_node(node_index)
        if type == 'node' and post_dom_tree.groups[num - id - 1].FindRoot().name != node_index:
            continue
        else:
            print(node_index, node_dict[node_index])
            file_write.writelines(node_dict[node_index] + '\n')
            file_write.writelines(value + '\n')
            file_write.writelines("\n")

    # print()
    # print(nodes[num - i - 1]['index'], post_dom_tree.tree_nodes[i].name, post_dom_tree.groups[i].name)


if __name__ == "__main__":
    file_name = 'encoder'
    # 从NiuTensor模型导出的网络结构文件中读取节点和常量
    NiuTensor_nodes, Constant_nodes = read_data(file_name)
    print(len(NiuTensor_nodes), len(Constant_nodes))

    nodes, constant_nodes = Init(NiuTensor_nodes, Constant_nodes)

    topo_graph = Graph()
    # node[0]是最终的输出，从node[0]开始遍历反向搜索，建立起逆序的dfs序列
    topo_graph.VisitExpr(nodes[0])
    # print(topo_graph.post_dfs_order)

    # 建立后序支配树
    post_dom_tree = DominatorTree()
    post_dom_tree.DominatorPartition(topo_graph)

    # 算子融合
    fuse_op_object = FuseOps()
    # 循环三次，分别进行三个融合规则
    for phase in range(0, 3):
        fuse_op_object.RunFuse(topo_graph, post_dom_tree, phase)

    # 打印融合后的group
    # print(len(post_dom_tree.groups))
    for node in post_dom_tree.groups:
        # if node.master_ref is not None:
        # print(node.name)
        logger.info(
            "[groups] {0} root:{1} node_num:{2}".format(node.name, node.FindRoot().name, node.FindRoot().num_nodes))

    SaveFusedGraph(open_file_name=file_name, save_file_name='fused_encoder')
