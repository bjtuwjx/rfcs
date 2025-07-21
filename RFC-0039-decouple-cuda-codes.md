<details>
<summary>Instructions - click to expand</summary>

- Fork the rfcs repo: https://github.com/pytorch/rfcs
- Copy `RFC-0000-template.md` to `RFC-00xx-my-feature.md`, or write your own open-ended proposal. Put care into the details.
- Submit a pull request titled `RFC-00xx-my-feature`. 
  - Assign the `draft` label while composing the RFC. You may find it easier to use a WYSIWYG editor (like Google Docs) when working with a few close collaborators; feel free to use whatever platform you like. Ideally this document is publicly visible and is linked to from the PR.
  - When opening the RFC for general discussion, copy your document into the `RFC-00xx-my-feature.md` file on the PR and assign the `commenting` label.
- Build consensus for your proposal, integrate feedback and revise it as needed, and summarize the outcome of the discussion via a [resolution template](https://github.com/pytorch/rfcs/blob/master/RFC-0000-template.md#resolution).
  - If the RFC is idle here (no activity for 2 weeks), assign the label `stalled` to the PR.
- Once the discussion has settled, assign a new label based on the level of support:
  - `accepted` if a decision has been made in the RFC
  - `draft` if the author needs to rework the RFC’s proposal
  - `shelved` if there are no plans to move ahead with the current RFC’s proposal. We want neither to think about evaluating the proposal
    nor about implementing the described feature until some time in the future.
- A state of `accepted` means that the core team has agreed in principle to the proposal, and it is ready for implementation. 
- The author (or any interested developer) should next open a tracking issue on Github corresponding to the RFC.
  - This tracking issue should contain the implementation next steps. Link to this tracking issue on the RFC (in the Resolution > Next Steps section)
- Once all relevant PRs are merged, the RFC’s status label can be finally updated to `closed`.

</details>

# Decouple CUDA Codes

**Authors:**

* @nickname
* @nickname 

## **Summary** （1人）王家喜

A short paragraph or bullet list that quickly explains what you're trying to do. <br />
目前，第三方硬件后端接入PyTorch的方式主要包括复用CUDA key和代码逻辑、利用树内预定义的key（如Intel XPU）和部分代码以及利用树内预留的PrivateUse1 key等三种。一方面，由于CUDA软件栈的生态地位，部分硬件厂商（如AMD HIP和MetaX MACA）选择直接复用CUDA key，通过兼容CUDA API的方式最小化PyTorch使用者的代码迁移成本。这种方法的优点是可以直接复用CUDA代码的逻辑，厂商适配工作量较小，但为了发挥硬件的优势，需要对CUDA kernel等代码进行侵入式修改。另一方面，随着PrivateUse1接入机制的不断完善，越来越多的厂商（如Ascend NPU和Cambricon MLU）选择此种接入方式，这种方法的优点是对PyTorch侵入修改较少，但厂商适配工作量较大（如无法直接复用CUDA代码逻辑）。<br />
本RFC提案旨在充分融合两者的优势，弥补相互之间的不足，先将CUDA代码解耦出来，形成相对独立的代码目录结构和编译单元；而后，逐步实现CUDA硬件后端、类CUDA硬件后端和其他架构硬件后端以统一的PrivateUse1机制接入PyTorch。

## **Highlights** （1人）袁孟雯

阐述CUDA代码分离工作的亮点

- 将 CUDA 相关实现从主工程中抽离，降低 PyTorch 核心框架对 CUDA 的直接耦合，提升整体工程可维护性。
- 更清晰、统一的目录层级结构，提升可读性与可维护性，使开发者能快速定位并理解后端逻辑，降低新开发者参与的学习门槛，为长期维护和社区贡献者提供更友好的结构。
- 重写构建系统以支持 CUDA 后端独立编译，降低编译复杂度，实现更快的增量构建和更少的构建依赖。
- 统一设备后端架构风格，为后续支持更多第三方后端提供模板，降低集成门槛和时间成本，提升 PyTorch 后端接入的一致性与可插拔性。

## **Motivation**（1人）祝贺

传统上，NVIDIA GPU与CUDA架构长期作为PyTorch生态中唯一的并行计算解决方案。随着越来越多的厂家推出自己的高效能计算设备，如寒武纪MLU、Graphcore IPU等，当前生态暴露出以下关键问题：
- 重复开发成本：各厂商独立开发设备适配层，导致重复编写
- 接口碎片化：不同硬件平台的API命名规则与实现方式差异显著，迫使用户维护多套设备专用代码。
- 操作复杂性：尽管部分厂商通过PrivateUse1机制实现基础接入，但设备管理语义与算子命名仍未统一
这种生态分裂现状与PyTorch硬件无关性的设计理念产生直接冲突，导致跨平台模型部署效率低下、硬件依赖性的研究复现困难、新型计算架构接入成本居高不下。

该方案的贡献：
- 抽象与标准化：对现有不同厂商的适配代码进行全面梳理，提取共性，将适配代码的命名和架构统一至PrivateUse1标准下，确保从前端到后端的一致性，减少不必要的重命名步骤。最终，让人工智能模型开发者在对设备无感的情况下使用pytorch。
- 代码复用与通用性提升：通过详细的调用栈分析，识别并抽象出如设备管理(device)、流(stream)管理、事件(event)处理等通用组件，形成一套统一的接口规范。这样，各厂商仅需关注实现这些通用接口的底层硬件特定逻辑，大幅降低适配成本和复杂度。
- 简化接入流程：建立一套标准化的接入流程指南，指导新加入的厂商如何快速、高效地基于PrivateUse1标准实现其硬件适配，确保新适配代码的高效整合与兼容性。
- 开源社区协作与生态建设：通过统一的适配模式，鼓励各厂商共享适配经验，促进技术交流，推动不同显卡生态在PyTorch框架中的成熟与发展，使得PyTorch不再是某几种GPU硬件设备的专利，而是成为高性能计算领域最通用的框架。

## **Proposed Implementation**

This is the bulk of the RFC. Explain the design in enough detail for somebody familiar with PyTorch to understand, and for somebody familiar with the implementation to implement. 
This should get into specifics and corner-cases, and include examples of how the feature is used, and how it will interact with other features. Any new terminology should be defined here.
Consider:

* using examples and diagrams to help illustrate your ideas.

* including code examples, if you're proposing an interface or system contract.

* linking to project briefs or wireframes that are relevant.

* 代码分离（1人）  张靖
  
  <h3 id="fc3655d2"><font style="color:rgba(0, 0, 0, 0.9);">一、需要解耦的功能模块</font></h3>

<font style="color:rgba(0, 0, 0, 0.9);">涉及到需要解耦的功能模块包括：</font>

```markdown
├── aten                    # Tensor ops相关代码
├── c10/cuda                # 设备管理核心代码
├── caffe2                  # torch和caffe2合并的遗留文件夹，当前仅需保留CMakeLists.txt，用于编译和连接aten和c10的源文件
├── torch/cuda              # Python端接口
├── torch/backends          # PyTorch硬件后端Python接口
├── torch/csrc/cuda         # 主要是Python端和C++端的pybind
├── torch/csrc/distributed  # distributed底层实现代码
├── torch/csrc/jit          # jit底层实现代码
├── torch/csrc/profiler     # profiler底层实现代码
├── torch/csrc/dynamo       # dynamo底层实现代码
└── torch/csrc/inductor     # inductor底层实现代码
```

<h3 id="d65ce516"><font style="color:rgba(0, 0, 0, 0.9);">二、解耦方式</font></h3>

<font style="color:rgba(0, 0, 0, 0.9);">我们通过实践总结了以下四种解耦方式，这四种方式并不是独立的，而是相互照应、相互补充的。</font>

<div style="text-align: center;">
    <img src="RFC-0039-assets/decoupling.png" alt="decoupling" style="width: 80%;">
    <p>图1 解耦方式</p>
</div>

<h4 id="83253c23"><font style="color:rgba(0, 0, 0, 0.9);">（一）文件间解耦</font></h4>

<font style="color:rgba(0, 0, 0, 0.9);">解耦</font><font style="color:rgba(0, 0, 0, 0.9);">标准如下：</font>

<h5 id="H6xQ7"><font style="color:rgba(0, 0, 0, 0.9);">1. 文件夹名称包含 </font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">cuda</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">cudnn</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">THC</font>`<font style="color:rgba(0, 0, 0, 0.9);"> 关键字。</font></h5>

<font style="color:rgba(0, 0, 0, 0.9);">示例：</font>

- `torch/backends/cuda`
- `torch/backends/cudnn`
- `torch/cuda`
- `aten/src/ATen/cuda`
- `aten/src/ATen/cudnn`
- `aten/src/ATen/native/cuda`
- `aten/src/ATen/native/cudnn`
- `aten/src/ATen/native/nested/cuda`
- `aten/src/ATen/native/quantized/cuda`
- `aten/src/ATen/native/quantized/cudnn`
- `aten/src/ATen/native/sparse/cuda`
- `aten/src/ATen/native/transformers/cuda`
- `aten/src/THC`
- `torch/csrc/cuda`
- `torch/csrc/distributed/c10d/cuda`

<h5 id="E2gdO"><font style="color:rgba(0, 0, 0, 0.9);">2. 文件名包含 </font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">cuda</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">cudnn</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">THC</font>`<font style="color:rgba(0, 0, 0, 0.9);"> 关键字。</font></h5>

<font style="color:rgba(0, 0, 0, 0.9);">示例：</font>

- `torch/csrc/distributed/rpc/tensorpipe_cuda.cpp`
- `torch/csrc/profiler/stubs/cuda.cpp`

<h5 id="bOOx4"><font style="color:rgba(0, 0, 0, 0.9);">3. 后缀名是 </font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">.cu</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">.cuh</font>`<font style="color:rgba(0, 0, 0, 0.9);">。</font></h5>

<font style="color:rgba(0, 0, 0, 0.9);">示例：</font>

- `torch/csrc/distributed/c10d/quantization/quantization_gpu.cu`

<h4 id="586d0dc4"><font style="color:rgba(0, 0, 0, 0.9);">（二）文件内解耦</font></h4>

有些cuda代码直接和torch代码耦合在一个文件内，通过环境变量、宏定义或者设备判断等隔离。

<h5 id="e819c231"><font style="color:rgba(0, 0, 0, 0.9);">1. 包含 CUDA 相关的环境变量判断</font></h5>

<font style="color:rgba(0, 0, 0, 0.9);">示例：</font>

```cpp
#if defined(__CUDA_ARCH__) 存在于下列文件
torch/csrc/aten/native/Distributions.h

#if defined(__CUDACC__) 存在于下列文件
torch/csrc/aten/native/sparse/Macros.h

#ifdef USE_CUDA 存在于下列文件或者文件夹
caffe2/CMakeLists.txt
torch/csrc/Storage.cpp
torch/csrc/dynamo/guards.cpp
torch/csrc/inductor/aoti_runner/pybind.cpp
torch/csrc/jit
```

<h5 id="f813fdd6"><font style="color:rgba(0, 0, 0, 0.9);">2. 文件内包含 CUDA 相关宏定义</font></h5>

+ `TORCH_CUDA_CU_API`
+ `TORCH_CUDA_CPP_API`
+ `TORCH_CUDA_CHECK`

<h5 id="2e31866b"><font style="color:rgba(0, 0, 0, 0.9);">3. 文件内包含 is_cuda、kCUDA、“cuda”等</font></h5>

示例：

```cpp
static CUDAHooksInterface* cuda_hooks = nullptr;
xxtensor.is_cuda()
xxtensor.device().type() == at::kCUDA
register_cuda_runner("cuda", &create_aoti_runner_cuda)
```

<h4 id="49e30103"><font style="color:rgba(0, 0, 0, 0.9);">（三）补充编译有依赖的文件</font></h4>

为了独立编译CUDA，CUDA编译需要依赖的文件也做了解耦。

<font style="color:rgba(0, 0, 0, 0.9);">需要补充的文件类型包括：</font>

<h5 id="FuDse">1. `<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">*.h</font>`<font style="color:rgba(0, 0, 0, 0.9);">、</font>`<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">*.hpp</font>`<font style="color:rgba(0, 0, 0, 0.9);"> 头文件。</font></h5>

示例：

<span class="path-highlight">torch/csrc/autograd/functions/comm.h</span>

<h5 id="x8Noj"><span class="header-highlight">2. 配置文件。</span></h5>

示例：

<span class="path-highlight">aten/src/ATen/ATenConfig.cmake.in</span>  
<span class="path-highlight">aten/src/ATen/Config.h.in</span>  
<span class="path-highlight">aten/src/ATen/native/native_functions.yaml</span>  
<span class="path-highlight">aten/src/ATen/native/tags.yaml</span>  
<span class="path-highlight">aten/src/ATen/native/ts_native_functions.yaml</span>

<h5 id="x6ttV"><span class="header-highlight">3. 模板文件。</span></h5>

示例：

<span class="path-highlight">aten/src/ATen/templates</span>

<h5 id="CTl4m"><span class="header-highlight">4. 打桩文件。</span></h5>

示例：

<span class="path-highlight">torch/csrc/stub.c</span>

<h4 id="b867b20f"><font style="color:rgba(0, 0, 0, 0.9);">（四）根据编译文件功能模块划分</font></h4>

<font style="color:rgba(0, 0, 0, 0.9);">有助于查漏补缺、去除冗余代码。</font>

示例 1：

<font style="color:rgba(0, 0, 0, 0.9);">通过</font>build_variables<font style="color:rgba(0, 0, 0, 0.9);">.bzl中文件划分解耦 distributed 模块 CUDA 相关代码</font>

```cpp
# These files are the only ones that are supported on Windows.
libtorch_cuda_distributed_base_sources = [
    "torch/csrc/distributed/c10d/reducer_cuda.cpp",
]

# These files are only supported on Linux (and others) but not on Windows.
libtorch_cuda_distributed_extra_sources = [
    "torch/csrc/distributed/c10d/CudaDMAConnectivity.cpp",
    "torch/csrc/distributed/c10d/NCCLUtils.cpp",
    "torch/csrc/distributed/c10d/FlightRecorder.cpp",
    "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp",
    "torch/csrc/distributed/c10d/ProcessGroupUCC.cpp",
    "torch/csrc/distributed/c10d/UCCTracing.cpp",
    "torch/csrc/distributed/c10d/UCCUtils.cpp",
    "torch/csrc/distributed/c10d/intra_node_comm.cpp",
    "torch/csrc/distributed/c10d/intra_node_comm.cu",
    "torch/csrc/distributed/c10d/CUDASymmetricMemory.cu",
    "torch/csrc/distributed/c10d/CUDASymmetricMemoryOps.cu",
    "torch/csrc/distributed/c10d/cuda/AsyncMM.cu",
    "torch/csrc/distributed/c10d/NanCheck.cu",
    "torch/csrc/distributed/rpc/tensorpipe_cuda.cpp",
    "torch/csrc/distributed/c10d/quantization/quantization_gpu.cu",
]

libtorch_cuda_distributed_sources = libtorch_cuda_distributed_base_sources + libtorch_cuda_distributed_extra_sources
```

示例 2：

根据aten\src\ATen\CMakeLists.txt中文件划分添加aten\src\ATen\native\miopen代码

```cpp
list(APPEND ATen_CUDA_CPP_SRCS
  ${cuda_cpp}
  ${native_cuda_cpp}
  ${native_cudnn_cpp}
  ${native_miopen_cpp}
  ${native_nested_cuda_cpp}
  ${native_quantized_cuda_cpp}
  ${native_quantized_cudnn_cpp}
  ${native_sparse_cuda_cpp}
  ${native_transformers_cuda_cpp}
)
```

* 目录重构（1人）  张靖
  
  <h3 id="7e0821a0"><font style="color:rgba(0, 0, 0, 0.9);">三、目录重构</font></h3>

<div style="text-align: center;">
    <img src="RFC-0039-assets/catalogue.png" alt="catalogue" style="width: 80%;">
    <p>图2 目录重构</p>
</div>

cuda解耦出来后，原始目录参考第一节，除了nvidia（cuda），我们调研了[AMD(gpu)](https://github.com/ROCm/pytorch)、[Google(TPU)](https://github.com/pytorch/xla/tree/master)、[Intel(XPU)](https://github.com/intel/intel-extension-for-pytorch)、[Ascend(NPU)](https://gitee.com/ascend/pytorch)、[Cambricon(MLU)](https://github.com/Cambricon/torch_mlu/tree/r2.4_develop)等多个超算卡厂商适配pytorch的方式，总结了各厂商适配PyTorch的代码目录结构、相似和特异性改动点，着重考虑到了以下因素：

<h5 id="Ee0MW">1. Python/C++分层解耦</h5>

通过物理隔离Python层（core/、backends/）和C++层（csrc/），明确区分接口定义与底层实现，降低代码耦合度。这样有助于Python层专注于业务逻辑和用户接口，而C++层则处理底层实现和性能优化。

<h5 id="CkxwH"><font style="color:rgb(44, 44, 54);">2. 模块化独立插件</font></h5>

将distributed/、profiler/作为独立插件，与核心框架解耦，使得各个模块可以独立开发、测试和迭代，同时也便于第三方开发者根据需要选择性地集成或扩展某些功能。

<h5 id="HowFi"><font style="color:rgb(44, 44, 54);">3. 统一硬件适配框架</font></h5>

合并 `c10/cuda` 和 `caffe2` 为 `framework/`，形成统一的设备管理与资源调度层，降低了维护成本。

<h5 id="CtkWI"><font style="color:rgb(44, 44, 54);">4. 目录重命名</font></h5>

<font style="color:rgb(44, 44, 54);">新的目录命名尽量直观地反映了其包含的内容和功能，例如 core 表示核心接口层，csrc 表示C++源代码，python 表示Python与C++的绑定层等，便于开发人员快速理解和导航项目代码。</font>

最后，整理出的新的适配代码目录结构如下：

```markdown
├── backends/                     # 只保留CUDA、cuDNN相关python接口
├── core/                         # Python核心接口层
├── csrc/                         # C++核心代码仓库
│   ├── python/                   # Python与C++绑定层
│   ├── aten/                     # Tensor、ops运算相关
│   ├── framework/                # 框架基础架构
│   ├── dynamo/                   # 动态图转静态图工具
│   ├── inductor/                 # 硬件代码生成与编译器模块
│   └── jit/                      # JIT编译器模块
├── distributed/                  # 分布式计算模块（独立插件）
└── profiler/                     # 性能分析模块（独立插件）
```

<font style="color:rgb(44, 44, 54);"></font>

<h3 id="7e0821a0"><font style="color:rgba(0, 0, 0, 0.9);">四、编译工程优化</font></h3>
本方案针对PyTorch原生CUDA设备编译流程进行了以下关键性改进：

- **编译逻辑解耦**  
   将CUDA编译系统从主框架解耦为独立工程，构建两大核心组件：
  
  - `torch_cuda`  
    ▸ 设备抽象层与运行框架  
    ▸ 设备资源管理  
    ▸ 算子实现（原生/native、加速库/cuBLAS/cuDNN/linalg、自定义）
  
  - `torch_python_cuda`  
    ▸ 基于pybind11的Python-C++交互接口  
    ▸ 针对新设备的跨语言类型系统桥接层，实现设备后端与Python层的双向解耦

- **CMake工程化封装**  
   基于`tools.setup_helpers.cmake`封装`wrapped_cmake`构建工具：
  
  - 标准化设备后端编译工具链
  - 实现：编译参数统一配置、环境自动初始化、编译器特性适配

- **模块化隔离架构**  
  
  - 分离出独立设备模块`_CUDAC.cpython-XX.so`，具备独立初始化链路
  - 统一新设备专用扩展构建器`torch.utils.cpp_extension.NewDeviceCppExtension`，实现编译环境与核心框架的物理隔离

                          ![编译架构对比](RFC-0039-assets/decouple_cuda_compiling_implementation.png)        
  _图4.1: 编译架构对比（左：原始架构，右：新架构）_

## 优缺点（1人）   付泽伟

## **Metrics **

理想情况下pytroch应该作为一种与硬件无关的深度学习框架，就像操作系统一样对于使用者屏蔽底层硬件实现细节，并提供经过抽象的和便于使用的接口，这些接口不应该涉及任何和底层硬件实现有关的信息。Pytorch自定义一套与底层硬件无关的硬件抽象层，统一差异化的硬件接口（集合通信），使上层系统组件无需关注具体硬件实现，同时方便各个硬件厂商对接自己的硬件。然而现实情况和上面有差异，主要是以下几点。

1. 直接指定底层硬件
   实际在使用pytorch的时候，经常涉及到在代码中直接指定底层硬件的情况，例如torch.tensor([3,4]).cuda()，假如在切换到第三方硬件后，pytorch的用户还需要对代码做不通程度的修改，而且由于缺乏硬件抽象，对于第三方的接入使用没有强制性的规定，导致用户代码在切换不同的底层硬件时所做的的修改不完全一样，给代码的通用性带来了挑战。
2. pytorch和cuda的强依赖
   pytorch源码中直接涉及到调用cuda的接口，这导致了新的cuda版本发布后，需要等pytorch官方适配，pytorch此外代码中充斥了对cuda头文件的引用，需要通过设置对应的环境变量加以屏蔽，不便于用户理解。
3. 第三方硬件接入困难
   目前pytorch提供了privateuse1的DispatechKey，为开发者提供了一种扩展硬件的方式，然后在厂商的实际使用中还是存在问题，例如1.无法同时接入两个不同的后端，2.代码的侵入性强，需要在Pytorch框架层面修改核心组件例如（storage模块，device manange模块），这导致与官方代码的耦合度高，而且无法跟随Pytorch的版本自动升级。
   我们提出的cuda代码抽象分离方案就是在看到以上问题的基础上提出的，主要具有以下的优点：
4. 对使用者屏蔽底层硬件实现
   我们自定义了一套对底层的硬件抽象层，规定了在接入第三方硬件时应该实现的接口和调用规则，在用户使用层面，用户不用直接使用cuda这样的关键字，我们自定义了一套通用的关键字（cuda对应pu1，nccl对应pccl），底层硬件改变后对用户是无感的，用户不用频繁修改代码，真正做到一套代码全平台运行。
5. 解除pytorch和cuda代码的强依赖
   我们将cuda设备视为一个和第三方硬件一样的可接入的硬件，对cuda设备的接入方式和所有第三方硬件一致，并从pytorch代码中删除了对cuda的依赖，这样pytorch的版本升级不用和cuda升级同步，给双方留下的最大的灵活性。
6. 方便接入第三方硬件
   以往的第三方硬件接入过程中，各个厂商分别实现接入代码，导致代码臃肿和功能重复，现在我们提供了硬件抽象层的基类实现，一些通用的功能已经实现完毕，并预留出了和硬件强相关的接口，各个厂商只需要按照要求实现这些接口即可实现硬件接入pytorch。由于通用了代码，当框架代码升级时第三方硬件也能自动享受框架升级带来的性能提升。

## **Drawbacks **

Are there any reasons why we should not do this? Here we aim to evaluate risk and check ourselves.

Please consider:

* is it a breaking change?
* Impact on UX
* implementation cost, both in terms of code size and complexity
* integration of this feature with other existing and planned features

## **Alternatives**   洪泓

What other designs have been considered? What is the impact of not doing this?

代码有以下两种放置方案：

1. in-tree

在Pytorch代码下新建目录pytorch/third_device/torch_cuda放入分离后代码，编译过程融       入Pytorch编译中，编译前通过patch形式对Pytorch原生代码进行修改，可以无缝集成到 PyTorch 生态系统中，和PyTorch进行同步开发和版本更新，安全性和稳定性更高，兼容性好，不需要再进行额外的代码适配和测试。

2. out-of-tree

不将代码直接集成到主代码库中，新建仓库对代码独立进行编译和维护，使用时以插件形式接入Pytorch，不对Pytorch原生代码进行侵入式修改，可以提高代码灵活性并降低代码维护成本，开发者可以在不影响主项目的情况下，自由地进行代码改进、修复漏洞和添加新功能，实现快速迭代和测试。

## **Prior Art**（1人） 崔巍

### 社区讨论
* Issue #129027（“将部分公用 API 从 torch::cuda::initModule 提取到 torch::initModule”）：讨论指出当前有一些仅与 CUDA 实现无关的公用功能（例如缓存张量开关、缓存张量管理等）被定义在 torch/csrc/cuda/Module.cpp 中解决思路是将这些与设备无关的 API 从 CUDA 初始化模块中抽取出来，放入通用的 torch 模块，以便其他设备也能复用
* Issue #131881（“解耦部分通用 API 与 CUDA 构建”）：与 #129027 类似，该提案关注将当前只能在启用 CUDA 时才暴露的通用 API（如 _set_storage_access_error_msg、_storage_Use_Count 等）移动到基础模块中问题描述中展示了一段 torch/csrc/cuda/Module.cpp 的注册代码片段，并建议“从 torch::cuda::initModule(module) 中移动到 torch::initModule()”，以实现功能与设备无关的解耦截至目前，这些讨论还处于提案阶段，主要体现了社区对将 CUDA 相关代码拆分为通用层的需求。

### 第三方厂商实践

* 寒武纪 (Cambricon)：寒武纪推出了名为 CATCH 的 PyTorch 扩展包，以支持 Cambricon MLU 设备CATCH 是独立于主 PyTorch 发行版的包，通过对原生 PyTorch 源码应用补丁（patch）的方式将寒武纪专用后端集成进去具体做法是在 PyTorch Extension 机制下，将 Cambricon 设备的 Aten 算子封装在 CATCH 中，然后借助补丁把这些算子注册到原生 PyTorch 的算子注册框架中Cambricon 的构建流程使用 CMake 脚本和 Docker 容器管理，CATCH 自身包含多卡训练和 TorchScript 图融合的支持，已实现对 MLU370 等硬件的训练/推理支持通过这种方式，PyTorch 编译时引入了 MLU 相关代码（如新增 MLU 设备类型和对应算子注册），在运行时可以选择使用 MLU 设备进行计算（类似 device="mlu"）。目前 CATCH 已能在开启补丁的 PyTorch 上支持多卡训练和 TorchScript 模式下的融合推理，但尚未形成通用动态多后端加载机制，通常需要使用特定配置的 PyTorch 二进制（带 MLU 补丁的版本）来运行。

* 摩尔线程 (MooreThreads)：摩尔线程提供了名为 torch_musa 的 PyTorch 插件包，通过 “插件化” 的方式支持其 MUSA GPU。该项目以扩展包形式发行，官方描述“以 plug-in 方式开发，使 torch_musa 与 PyTorch 解耦”实现原理是利用 PyTorch 的 PrivateUse1 设备键（PrivateUse1 预留给第三方硬件）注册“MUSA”设备类型，并通过文本转换和自定义编译工具链将 CUDA 代码适配到 MUSA。具体包括使用自研的 MUSAExtension（类似于 CUDAExtension）来构建本地扩展、使用 SimplePorting 等工具将 .cu 文件中的 cuda 替换为 musa、将依赖如 cublas 替换为 MUSA 对应的库等构建时需依赖摩尔线程提供的 MUSA 编译器（mcc）和 SDK，并可通过其脚本自动下载并编译改造后的 PyTorch 和 torch_musa。使用 torch_musa 后，用户可以像使用 CUDA 一样使用 MUSA 设备（相同的 API 调用格式），且兼容原生 PyTorch 的编程习惯实践效果方面，torch_musa 提供了对 MUSA GPU 的张量计算支持，并且声明了“实现了 CUDA 兼容性，大大减少新算子适配的工作量”目前 torch_musa 已有多个版本的轮子和源码发布，支持在不修改上层模型代码的前提下使用 MUSA 设备进行训练和推理；动态多后端切换方面，可通过设置不同的 torch.device("cuda") 或 torch.device("musa") 来选择对应硬件，但底层需要先行安装并加载相应插件版本的 PyTorch。

- 总体而言，Cambricon 和摩尔线程都通过插件式、补丁式改造方式实现了 CUDA 编译逻辑的拆分：前者需要维护带补丁的 PyTorch 分支，后者则在保持主 PyTorch 源兼容的基础上提供独立扩展包，两者都在实践中支持了各自设备的动态加载与调用。

## **How we teach this**

* What names and terminology work best for these concepts and why? How is this idea best presented?
* Would the acceptance of this proposal mean the PyTorch documentation must be re-organized or altered?
* How should this feature be taught to existing PyTorch users?

## **Unresolved questions**

* What parts of the design do you expect to resolve through the RFC process before this gets merged?
* What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
* What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?

## Resolution

We decided to do it. X% of the engineering team actively approved of this change.

### Level of Support

Choose one of the following:

* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.

#### Additional Context

Some people were in favor of it, but some people didn’t want it for project X.

### Next Steps（1人）侯丽亚

Will implement it. 

Phase 1: 代码解耦与目录重构 (预计周期: 2个月)

1. 核心模块解耦
  - 完成 `aten/src/ATen/cuda` 和 `c10/cuda` 的代码分离，建立独立编译单元
  - 重构 `torch/csrc/cuda` 的 Python-C++ 绑定层，确保与核心框架解耦
  - 验证分布式 (`distributed/c10d/cuda`) 和性能分析 (`profiler/stubs/cuda`) 模块的插件化可行性
2. 目录结构调整
  - 迁移 CUDA 相关代码至新目录结构（如 `csrc/framework/cuda`）
  - 标准化 backends/ 下的 Python 接口，统一命名规范（如 `torch.backends.pu1` 替代 `torch.backends.cuda`）
3. 构建系统适配
  - 实现 torch_cuda 和 torch_python_cuda 的独立 CMake 工程
  - 统一新设备专用扩展构建器 `torch.utils.cpp_extension.NewDeviceCppExtension` ，支持多后端编译隔离

Phase 2: 兼容性测试与第三方硬件接入 (预计周期: 1个月)

1. 向后兼容性保障
  - 维护 torch.cuda.* 的临时别名，通过 Deprecation Warning 引导用户迁移至 torch.pu1.*
  - 测试现有 CUDA 模型的兼容性（重点验证 is_cuda() 等调用的替换逻辑）
  - 编写《硬件后端接入指南》
2. 第三方厂商协作
  - 与 moer 等厂商合作，验证 PrivateUse1 接入路径的可行性
  - 编写《硬件后端接入指南》

#### Tracking issue

<github issue URL>

* 主跟踪任务: #[GitHub Issue Number]
* 子任务拆分:
  - 代码解耦: #[Sub-Issue 1]
  - 构建系统: #[Sub-Issue 2]
  - 文档更新: #[Sub-Issue 3]

#### Exceptions

Not implementing on project X now. Will revisit the decision in 1 year.
