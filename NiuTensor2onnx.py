import onnx
import onnx.helper as helper
from onnx import numpy_helper
import re
import numpy as np
import logging
from onnx import shape_inference, TensorProto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[NiuTensor2onnx]")

NiuTensor_OPPATTERN = {'L_CROSSENTROPY': 'CrossEntropy',
                       'M_GATHER': 'Gather',
                       'F_RECTIFY': 'Recify',
                       'M_MULTIPLY_I': 'Mul',
                       'M_MULTIPLYDIM': 'MultiplyDim',
                       'M_SUMDIM': 'SumDim',
                       'M_MATRIXMUL': 'Gemm',
                       'R_REDUCEVARIANCE': ['ReduceMean', 'Sub', 'ReduceSumSquare'],
                       'M_MULTIPLY': 'Mul',
                       'S_SPLIT': 'Split',
                       'S_UNSQUEEZE': 'Unsqueeze',
                       'F_SOFTMAX': 'Softmax',
                       'M_OPERATION': 'Operation',
                       'M_SUB': 'Sub',
                       'R_REDUCEMEAN': 'ReduceMean',
                       'M_MATRIXMULBATCHED': 'MatrixMulBatched',
                       'M_POWER': 'Pow',
                       'M_SCALE': 'Scale',
                       'M_SUM': 'Sum',
                       'S_MERGE': 'Squeeze',
                       'M_DIV': 'Div'
                       }

onnx_DTYPE = {
    0: TensorProto.FLOAT,
    1: TensorProto.FLOAT16,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL
}

numpy_DTYPE = {
    0: np.float32,
    6: np.int32
}

NiuTensorDTYPE_2_ONNXDTYPE = {
    "X_FLOAT": 0,
    "X_INT": 6
}


class NiuTensor_node:
    def __init__(self):
        self.index = -1
        self.incomenum = 0
        self.outgonum = 0
        self.income = []
        self.operator = ""
        self.outgo = []
        self.order = 0
        self.dimsize = []
        self.dtype = NiuTensorDTYPE_2_ONNXDTYPE['X_FLOAT']


class Constant_node:
    def __init__(self):
        self.index = -1
        self.outputnum = 0
        self.output = []
        self.order = 0
        self.dimsize = []
        self.dtype = NiuTensorDTYPE_2_ONNXDTYPE['X_FLOAT']


def not_empty(s):
    if s == '-' or s == ',' or s == 'node' or s == '':
        return False
    return True


def read_data(file_name):
    NiuTensor_nodes = []
    Constant_nodes = []
    file = open(file_name, "r")
    text = file.read()
    text = text.split('\n')
    text = list(filter(None, text))
    for i in range(0, len(text), 2):
        node = NiuTensor_node()
        info = text[i]
        value = text[i + 1]
        info = info.split(' ')
        info = list(filter(not_empty, info))
        # print(info)
        cnt = 0
        node.index = info[cnt]
        cnt += 1
        match = re.match(r'income(.)(\d+)(\S+)', info[cnt])
        cnt += 1
        if match:
            node.incomenum = match.group(2)
        else:
            logger.error('match incomenum error , exit!')
            exit(1)
        if int(node.incomenum) != 0:
            match = re.match(r'(\S+)(.)(.)', info[cnt])
            cnt += 1
            if match:
                # print(match.group(1))
                node.operator = match.group(1)
            else:
                logger.error('match operator error , exit!')
                exit(1)
            for i in range(0, int(node.incomenum)):
                node.income.append(info[cnt])
                cnt += 1
            # print(node.income)
        else:
            cnt += 1
        # print(info[cnt])
        match = re.match(r'outgo(.)(\d+)(\S+)', info[cnt])
        cnt += 1
        if match:
            node.outgonum = match.group(2)
        else:
            logger.error('match outgonum error , exit!')
            exit(1)
        for i in range(0, int(node.outgonum)):
            node.outgo.append(info[cnt])
            cnt += 1
        value = re.split(' |=|,', value)
        # print(value)
        node.order = value[1]
        for i in range(0, int(node.order)):
            node.dimsize.append(int(value[3 + i]))
        # print(node.dimsize)
        node.dtype = NiuTensorDTYPE_2_ONNXDTYPE[value[-3]]
        # print(node.dtype)
        if int(node.incomenum) != 0:
            NiuTensor_nodes.append(node)
        else:
            Constant_nodes.append(node)
    return NiuTensor_nodes, Constant_nodes


def To_onnx(file_name, NiuTensor_nodes, Constant_nodes):
    '''
        make_node [类型:NodeProto]make_node(op_type,inputs,outputs,name=None,doc_string=None,**kwargs)
        op_type:节点的算子类型 [类型:字符串]
        比如Conv、Relu、Add这类，详细可以参考onnx给出的算子列表，这个可以自己赋值，但最好与官网对应上，否则其他框架在跑onnx的时候会不知道这是什么。
        inputs:存放节点输入的名字 [类型:字符串列表]
        每个节点输入的数量根据情况会有不同，比如inputs(2-3)，即输入为2个或3个，可选的输入都会标注(optional)。以Conv为例，必有输入X和权重W，偏置B作为可选。
        outputs:存放节点输出的名字 [类型:字符串列表]
        与inputs类似，同样需要根据官网给出的输出个数来设置，大多数情况是一个输出，我暂且还没碰到多输出情况。

        name:节点名，可有可无，不要和op_type搞混了
        doc_string:描述文档的字符串，这个默认为None [类型:字符串]
        kwargs:存放节点的属性attributes [类型:任意]
        '''
    onnx_nodes = []
    for node in NiuTensor_nodes:
        node_def = helper.make_node(
            node.operator,
            node.income,
            [node.index],
            # node.index
        )
        onnx_nodes.append(node_def)
    '''
    make_tensor_value_info [类型:ValueInfoProto]make_tensor_value_info(name,elem_type,shape,doc_string="",shape_denotation=None)
    name:数据信息名字 [类型:字符串]
    elem_type:数据类型 [类型:TensorProto.DataType]
    shape:数据维度(形状) [类型:int列表/元组]

    doc_string:描述文档的字符串，这个默认为None [类型:字符串]
    shape_denotation:这个没太看懂，可能是对shape的描述 [类型:字符串列表]
    根据数据类型和形状创建一个ValueInfoProto。
    '''
    print(NiuTensor_nodes[0].index, NiuTensor_nodes[-1].index)
    output_node_def = helper.make_tensor_value_info(
        # NiuTensor_nodes[0].index,
        NiuTensor_nodes[0].index,
        onnx_DTYPE[NiuTensor_nodes[0].dtype],
        NiuTensor_nodes[0].dimsize
    )
    output_nodes = [output_node_def]

    # print(NiuTensor_nodes[-1].income)
    input_nodes = []
    # for node in Constant_nodes:
    #     for i in range(0, int(node.outgonum)):
    #         print(node.outgo[i])
    #         if node.outgo[i] in NiuTensor_nodes[-1].income:
    #             input_node_def = helper.make_tensor_value_info(
    #                 node.index,
    #                 onnx_DTYPE[NiuTensor_nodes[-1].dtype],
    #                 NiuTensor_nodes[-1].dimsize
    #             )
    #             input_nodes.append(input_node_def)
    #             break

    for node in Constant_nodes:
        input_node_def = helper.make_tensor_value_info(
            node.index,
            onnx_DTYPE[node.dtype],
            node.dimsize
        )
        input_nodes.append(input_node_def)
    print(len(input_nodes))
    '''
    make_tensor [类型:TensorProto]make_tensor(name,data_type,dims,vals,raw=False)
    name:数据名字，要与该数据的信息tensor value info中名字对应 [类型:字符串]
    data_type:数据类型 [类型:TensorProto.DataType] 如TensorProto.FLOAT、TensorProto.UINT8、TensorProto.FLOAT16等
    dims:数据维度 [类型:int列表/元组]
    vals:数据值，好像要可迭代的 [类型:列表/任意]

    raw:选择是否用二进制编码 [类型:bool]
    raw为False的时候，就会用相应的TensorProto来存储基于data_type的值，若raw为True，则是用二进制编码来存储数据。
    '''
    onnx_initializer = []
    for node in Constant_nodes:
        # print(node.dimsize)
        # print(node.dtype)
        # data = np.empty(node.dimsize, dtype=numpy_DTYPE[node.dtype])
        # # tensor = onnx.numpy_helper.from_array(data)
        # data = data.tolist()
        # print(len(data))
        size = 1
        for i in node.dimsize:
            size *= i
        data = list(0 for i in range(0, size))
        initializer_def = helper.make_tensor(
            node.index,
            onnx_DTYPE[node.dtype],
            node.dimsize,
            data
        )
        onnx_initializer.append(initializer_def)

    '''
    make_graph [类型:GraphProto]make_graph(nodes,name,inputs,outputs,initializer=None,doc_string=None,value_info=[])
    nodes:用make_node生成的节点列表 [类型:NodeProto列表]
    比如[node1,node2,node3,…]这种的
    name:graph的名字 [类型:字符串]
    inputs:存放graph的输入数据信息 [类型:ValueInfoProto列表]
    输入数据的信息以ValueInfoProto的形式存储，会用到make_tensor_value_info，来将输入数据的名字、数据类型、形状(维度)给记录下来。
    outputs:存放graph的输出数据信息 [类型:ValueInfoProto列表]
    与inputs相同。

    initializer:存放超参数 [类型:TensorProto列表]
    比如Conv的权重W、偏置B，BatchNormalization的scale、B、mean、var。这些参数数据都是通过make_tensor来转换成TensorProto形式。
    doc_string:描述文档的字符串，这个默认为None [类型:字符串]
    value_info:存放中间层产生的输出数据的信息 [类型:ValueInfoProto列表]
    '''
    # print(len(onnx_initializer))
    graph_def = helper.make_graph(onnx_nodes,
                                  name='onnx_graph',
                                  inputs=input_nodes,
                                  outputs=output_nodes,
                                  initializer=onnx_initializer
                                  )

    model_def = helper.make_model(graph_def, producer_name='NiuTensor2onnx')
    # helper.printable_graph(model_def.graph)
    onnx.save(model_def, file_name)


if __name__ == "__main__":
    nodes, initializer = read_data('encoder')
    To_onnx('encoder.onnx', nodes, initializer)
    # print(len(nodes))
    # print(len(Constant_nodes))
