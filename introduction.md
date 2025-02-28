# 什么是ONNX？
开放神经网络交换（Open Neural Network Exchange）简称ONNX是微软和Facebook提出用来表示深度学习模型的开放格式。所谓开放就是ONNX定义了一组和环境，平台均无关的标准格式，来增强各种AI模型的可交互性。
## 为什么提出ONNX？
随着深度学习框架（如TensorFlow、PyTorch、Caffe等）的快速发展，各种框架之间往往不兼容，导致用户在开发和部署模型时需要面对不同平台之间的迁移和转化问题。为了避免过多的重复工作和框架锁定（即依赖某个特定框架的格式和操作），ONNX应运而生。它旨在为深度学习模型提供一个统一的标准，帮助模型在不同的框架和硬件之间自由流动。
## ONNX的作用

1. 跨框架兼容性：ONNX支持将训练好的模型从一个框架（如PyTorch）转换为另一个框架（如TensorFlow）。这使得开发者能够选择最适合自己需求的框架，并轻松转换为其他框架，避免被单一框架锁定。

2. 加速部署：ONNX支持多种硬件平台，包括CPU、GPU和一些专用加速硬件（如TPU、FPGA等），通过标准化模型表示，简化了模型在不同硬件平台上的优化和部署。

3. 提高性能：通过将模型转换为ONNX格式，可以使用ONNX的优化工具（如ONNX Runtime）来优化模型性能，减少计算开销，提高推理效率。

4. 模型互操作性：ONNX使得不同工具和库能够一起工作，促进了AI生态系统的合作和发展。它允许开发者使用更广泛的工具链（如NVIDIA TensorRT、Intel OpenVINO等）来进行模型加速和优化


# ProtoBUF简介
在分析ONNX组织格式前，我们需要了解一下ProtoBuf,因为ONNX是使用protobuf这个序列化数据结构存储神经网络的权重信息（本人本来并不知道好多框架（ONNX；openppl的PMX格式）是基于ProtoBuf序列化格式去存储的，学习自B站UP主ZOMI酱）。

## ProtoBuf的主要特点
1. 高效性：

   * Protobuf采用二进制格式进行数据序列化，相比XML和JSON等文本格式，它能提供更小的文件体积和更高的解析/序列化速度。这使得它特别适合用于网络传输和存储。
2. 跨语言支持：
   * Protobuf支持多种编程语言，包括C++, Java, Python, Go, Ruby, JavaScript, C#, Objective-C等。通过定义 .proto 文件，开发者可以在不同的编程语言之间共享数据结构。
3. 向后兼容性：
   * Protobuf允许在不破坏现有数据结构的情况下，添加新的字段或更改现有字段。这种向后兼容性使得应用在数据格式变化时不容易受到影响，支持版本控制。
4. 灵活性：
    * 通过 .proto 文件定义数据结构，可以自定义不同类型的数据字段。protobuf不仅支持基本的类型（如整数、浮点数、字符串等），还支持复杂的嵌套数据类型、数组、枚举等。
5. 高效的序列化和反序列化：
    * protobuf在序列化和反序列化方面非常高效，尤其适用于大规模数据传输和存储，能够极大地提高应用程序的性能。

## ProtoBuf的工作原理
1. 定义数据结构（.proto 文件）： 使用 Protobuf 时，首先需要定义数据结构，这些结构被写在 .proto 文件中，里面包含了字段的名字、类型和标签（字段号）
    ```
    syntax = "proto3";  # 用 proto3 语法规则来解析和生成相应的代码

    message Person {
    string name = 1;
    int32 id = 2;
    string email = 3;
    }

    ```
    在上面的例子中，Person 是一个数据结构，它包含了三个字段：name（字符串类型）、id（整型）、email（字符串类型）。每个字段都有一个唯一的编号（例如，name = 1）。这些编号在序列化时会使用，而不依赖字段名，因此序列化数据的体积非常小。

2. 编译 .proto 文件： Protobuf的编译器（protoc）将 .proto 文件编译成特定编程语言的代码（如C++、Python、Java等）。这些代码包含了用于序列化和反序列化数据的类和方法。
    ```
    protoc --python_out=. addressbook.proto 
    ```
    会将 .proto 文件转换成Python代码，生成相应的类和方法
3. 序列化和反序列化
    * 序列化：将数据结构转化为二进制格式，方便存储或传输。
    * 反序列化：将二进制数据恢复为原来的数据结构。

# ONNX结构分析
加载一个 ONNX 模型文件后，我们会得到一个 ModelProto 对象。ModelProto 是 ONNX 模型的顶层容器，它主要包含以下几类信息：
1. 生产者信息（Producer Information）
    * producer_name：表示生成该模型的框架或工具的名称（例如：PyTorch、TensorFlow、ONNX等）。
    * producer_version：表示生产该模型的工具的版本号（例如：1.8.0）。
    * domain：表示该模型所属的领域（如：ai.onnx）。
    * description：对模型的描述或生成该模型的其他信息（例如：转换过程的描述）
2. GraphProto（图信息）
    * node：图中所有节点的列表，每个节点对应一个操作（例如卷积、全连接等）。每个节点都有操作类型、输入和输出信息。
    * input：图的输入节点，定义了模型输入的张量（如图像、文本等）。
    * output：图的输出节点，定义了模型的输出张量（如分类结果、预测值等）。
    * initializer：模型的权重参数（如卷积核、全连接层的权重），这些是与计算图相关的参数张量。
    * name：模型的名称，便于区分不同的模型
3. 模型元数据（Metadata）

    ONNX 模型的元数据部分包含了可选的其他信息，用于描述和标识模型：
    * metadata_props：元数据属性，通常包括与模型相关的其他详细信息，如开发者、使用的训练数据集等。
    * model_version：模型的版本号（例如：1.0）。
    * doc_string：对模型的文档说明或描述，通常用于描述模型的用途、背景或其他重要信息。
  
4. 训练/推理元数据（Training/Inference Info）
    * opset_import：这个字段包含了模型使用的操作集（opset）。不同版本的 ONNX 操作集定义了支持的操作类型、参数等。
    * graph：这是核心部分，即计算图（GraphProto）。它包含了计算图的节点、输入、输出、权重等详细信息。
5. 其他

大概的示意图如下：
```
message ModelProto {
  string ir_version = 1;                 // ONNX IR 版本
  string producer_name = 2;              // 生成该模型的工具名称
  string producer_version = 3;           // 工具版本
  string domain = 4;                     // 模型领域
  string description = 5;                // 模型描述
  repeated MetadataProps metadata_props = 6;  // 元数据
  repeated GraphProto graph = 7;         // 计算图信息
  repeated OpsetImport opset_import = 8; // 操作集版本
  string model_version = 9;              // 模型版本
  string doc_string = 10;                // 文档说明
}

```
# ONNX重要成分
## onnx.helper
onnx.helper 是 ONNX 库提供的一个辅助工具集合，帮助用户更轻松地创建和操作 ONNX 模型。它简化了模型的构建和序列化操作，尤其是在手动创建或修改 ONNX 模型时。通过 onnx.helper，您可以方便地创建节点、图、张量等。

常用的 helper 方法包括：
1. onnx.helper.make_model()：构建一个完整的模型。
2. onnx.helper.make_node()：创建一个操作节点（例如卷积、全连接等）。
3. onnx.helper.make_tensor()：创建一个张量（例如权重）。
4. onnx.helper.make_graph()：创建一个计算图。

例如，使用 onnx.helper 创建一个简单的计算图：
```
import onnx
from onnx import helper, TensorProto

# 创建节点
node = helper.make_node(
    'Add',  # 操作类型
    inputs=['x', 'y'],  # 输入张量
    outputs=['z'],  # 输出张量
)

# 创建图
graph = helper.make_graph(
    [node],  # 节点列表
    'simple_graph',  # 图名称
    [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2]),  # 输入
     helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])],
    [helper.make_tensor_value_info('z', TensorProto.FLOAT, [1, 2])],  # 输出
)

# 创建模型
model = helper.make_model(graph)
```
## TensorProto
TensorProto 用于表示 ONNX 模型中的张量。张量是神经网络中的基本数据单位，包括输入数据、权重参数、输出结果等。

TensorProto 通常包含以下信息：
1. name：张量的名称。
2. data_type：张量的数据类型（如 FLOAT, INT32, UINT8 等）。
3. dims：张量的形状，通常是一个整数列表（例如 [1, 3, 224, 224]）。
4. raw_data：张量的实际数据，以字节流的形式存储，通常用于存储权重。
   
例如，创建一个张量：
```
from onnx import helper
from onnx import TensorProto
import numpy as np

tensor = helper.make_tensor(
    name="weight",
    data_type=TensorProto.FLOAT,
    dims=[3, 3],
    vals=np.random.rand(9).astype(np.float32).tolist()  # 随机初始化一个权重矩阵
)
```
## GraphProto
GraphProto 是 ONNX 模型的核心部分，包含了计算图的详细信息。它包括了模型的所有节点、输入、输出、初始权重等信息。GraphProto 主要包含以下字段：

1. node：表示图中的操作节点，每个节点代表一个运算（如卷积、加法、矩阵乘法等）。
2. input 和 output：表示计算图的输入和输出张量。
3. initializer：包含了所有与模型参数（如权重、偏置）相关的张量。
4. name：图的名称。
5. doc_string：对图的描述。

## Operator (Op)
Operator 是 ONNX 中的一种抽象，表示计算图中的基本操作。每个操作有一个名称（如 Conv, MatMul）和一组输入输出。
1. 在 ModelProto 中，每个节点（node）都代表一个操作符，节点的 op_type 指定了操作类型（例如卷积、加法等）。
2. OpType 是 ONNX 模型定义中非常重要的一部分，它决定了如何处理输入并生成输出。
## Tensor
ONNX 中的 TensorProto 主要用于表示模型中的数据和权重张量。它通过 initializer 来存储权重，或者通过 input 和 output 来存储数据。
## Attribute
每个节点（node）都有一组 AttributeProto，这些属性描述了节点操作的特定参数。例如，卷积操作可能具有如下属性：滤波器大小、步幅、填充等。

