# Techniques for Large Scale Training

## Theory

Circa 2021: [How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/)

Circa 2022: [Techniques for training large neural networks](https://openai.com/index/techniques-for-training-large-neural-networks/)

Circa 2023: [Some Techniques To Make Your PyTorch Models Train (Much) Faster](https://sebastianraschka.com/blog/2023/pytorch-faster.html)

### Measuring Model Size

The first step in training a large scale model is to estimate the amount of memory you model will take up. This will then inform us as to which of the techniques below we will need to use. For example, FairScale has a really good flowchart on the decisions you need to make when training a large model (their flow chart is for whether or not to use FairScale, but you can apply it equally well to which techniques to use in training):

![Training Flowchart](https://fairscale.readthedocs.io/en/stable/_images/flowchart.png)

For example, if you model fits on a single GPU, then chances are Distributed Data Parallel is going to be sufficient to train your model. But this all hinges on knowing how much memory your model will take up.

There are a number of libraries to help you do this, including [torchinfo]() and [Accelerate estimate-memory](https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/estimate.py#L285)([docs](https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator)). There is an excellent blog post at [EleutherAI](https://blog.eleuther.ai/transformer-math/) which goes into more detail about estimating model sizes which I highly recommend. 

In short, we cannot just look at the number of parameters in a model to let us know the amount of memory (specifically VRAM) the model will take up. For example, let's assume we have a model with $P$ parameters, and assume this model will be trained in fp32 precision. Since gp32 contains 4 bytes, a naive estimate of the model memory in bytes would be $m = 4*P$. There is always going to be a small abount of overhead as well, and a conservative estimate for this additional overhead is < 20% of the model memory used. However, if we use mixed precision training, we need to keep an additional 4 bytes/param copy of the model in our optimizer states to help us maintain fidelity.

Mixed Precision Base Model Size: $2 bytes/param * P + 4 bytes/param * P$

However, pure parameter memory is not the only source of VRAM usage during training. We also have to take into account the storage of optimizer states and gradients in the device memory. For a basic optimizer like Adam, we need to keep an additional 3 copies of each parameter for the gradient calculations. Why? AdamW for example uses 4 bytes/param each for calculating momentum and variance, and an additional 4 bytes/param for a copy of all the parameters. SGD for example has to track momentum for each parameters, so it contains a 4 bytes/param for each parameter and 4 bytes/param for momentum, yielding 2 copies of each param in the model.

AdamW + Mixed Precision: $(2 + 4 + 12) bytes/param * P$

We also need to store the gradients themselves, which is an additional 2 or 4 bytes/param depending on the training precision.

Gradients + AdamW + MixedPrecision: $(2 + 4 + 12 + 2) bytes/param * P$

Finally, in calculating the gradients, we also need to store the activations at each layer, which are used in the gradient calculation. We can use tricks like activation recomputation/checkpointing to reduce this memory usage, but in general, the amount of memory needed to store the activation layers in fp16 precision is:

Activation Memory = $sbhL (\frac{24}{t} + 5\frac{a*s}{h*t})$ bytes

where:
- $s$ is the sequence length, in tokens
- $b$ is the batch size per GPU
- $h$ is the dimension of the hidden size within each transformer layer
- $L$ is the number of layers in the transformer model
- $a$ is the number of attention heads in the transformer model
- $t$ is the degree of tensor parallelism being used (1 if not)
- We assume no sequence parallelism is being used
- We assume that activations are stored in fp16

The activation memory typically is the largest factor during training, as we can see in this graph from Megatron:

![Megatron](https://blog.eleuther.ai/images/blog/transformer-math/activations.png#center)

So in general, the basic memory calculation for training a model is:

`Total Memory = Model Memory + Optimizer Memory + Activation Memory + Gradient Memory`

Let's take a simple example of the original GPT model from [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), which was a 12 layer decoder only transformer, which an inner dimension of 768, a context length of 512, and 12 attention heads. In the implementation in our repository (including bias in each layer and a vocab size of 40512) this yields ~147m parameters.

This yields:

Model Memory = $147m * 2 bytes/param = 294m$ bytes
Optimizer Memory = $147m * (4 + 12) bytes/param = 2352m$ bytes
Activation Memory = $512*768*12*(10 + 24 + 5\frac{12*512}{768}) = 349175808$ bytes * batch_size
Gradient Memory = $147m * 2 = 294m$ bytes

Total Memory = $294m + 2352m + B*349m + 294m = 2940m + B*349m$

With a batch size of 64, this yields ~25.3G of VRAM used, which should fit comfortably on an A100 with 40 GB of memory. Factoring in 20% overhead yields around ~30.3G of VRAM used in training, which matches empirically what we have found in training this model on a 40GB A100.

#### References

[Transformer Math 101](https://blog.eleuther.ai/transformer-math/)
[The FLOPs Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)

### Mixed Precision

Mixed precision training is a training speed optimization, at the expense of increased VRAM usage. The idea is to train models using lower-precision numbers, commonly fp16 or bf16. Modern accelerators can reach much higher FLOP counts on lower-precision numbers, so training speed can be improved due to the higher throughput. In addition, you save VRAM in inference, since you are using typically half the precision over fp32 in storing model parameters. However, due to the reduced precision during training, you typically need to store a full precision copy of the model weights in the optimizer states, so that you don't have gradient calculation issues.

![Mixed Precision Training Flow](https://lilianweng.github.io/posts/2021-09-25-train-large/mixed-precision-training.png)

In order to make mixed precision training work, there are three main architectural changes under the covers

- *Full-precision master copy of weights*: You need to maintain a full-precision (fp32) master copy of the weights that acucmulate gradients. These numbers are rounded up to half-precision for calculating the forward and backward passes. The intuition here is that each gradient update might be too small to be fully contained in the half-precision range.
- *Loss scaling*: Scale up the loss to better handle gradients with a small magnitude.
- *Arithmetic Precision*: For common model arithmetic (e.g. vector dot-product, element-wise summation, etc), accumulate the partial results in full-precision and then save the final output in memory in half-precision.

Pros:
    - Increased training speed (FLOP throughput).
    - Reduced VRAM usage in inference

Cons:
    - Increased VRAM usage in training over fp32 precision.

The HuggingFace [Accelerate](https://github.com/huggingface/accelerate) library can handle mixed precision training using [AMP](https://developer.nvidia.com/automatic-mixed-precision) seamlessly for you under the covers.

#### References

[Mixed Precision Training](https://arxiv.org/abs/1710.03740)
[AMP](https://developer.nvidia.com/automatic-mixed-precision)
[Mixed Precision Training Blog](https://lilianweng.github.io/posts/2021-09-25-train-large/#mixed-precision-training)

### Gradient Accumulation

Gradient accumulation is a technique to achieve a larger effective batch size in training than is possible given the memory constraints of the machine being used for training. Instead of updating the model parameters after each mini-batch, the gradients for the model parameters are accumulated for N steps. At the end of N steps, the backward pass is calculated using the accumulated gradients, effectively updating the model as if the training batch size were B*N.

There is some conflicting evidence on the utility of gradient accumulation. Some models, especially models trained with contrastive losses, benefit greatly from larger batch sizes. In this case, gradient accumulation can help generate these larger effective batch sizes. However, the evidence for this is limited, and typically gradient accumulation is useful when VRAM is extremely limited and you want to train with a "reasonable" effective batch size. Research suggests the variance of the gradients is a decreasing function of the batch size, so this suggests that using gradient accumulation to generate a larger effective batch size can yields less noisy gradients and a smoother learning regime.

Pros:
    - *Memory efficiency*: Allows training with effective batch sizes larger than fit in VRAM.
    - *Network efficiency*: Reduces the communication overhead in sharing gradients across all workers, since the allreduce only happens every N steps.
    - *Stable training*: Larger batch sizes can yield lower variance in the gradient updates.
    - *Improved Generalization*: For certain model types (e.g. constrastive models) a larger batch size yields better generalization,

Cons:
    - Negligible benefit if the desired batch fits in memory

#### References

[Hugging Face Tutorial](https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation)
[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
[Revisiting Small Batch Training for Deep Neural Networks](https://arxiv.org/abs//1804.07612)
[The Impact of the Mini-batch Size on the Variance of Gradients in Stochastic Gradient Descent](https://arxiv.org/abs/2004.13146)

### Gradient Checkpointing

Gradient checkpointing (sometimes called activation checkpointing, activation recomputation, or selective activation recomputation) is a technique to reduce the training memory footprint at the cost of increased computation (generally one additional forward pass per batch). The key insight is that to compute the gradients in the backwards pass, you need to save the activations in each layer from the forward pass, which can consume a lot of VRAM. Gradient checkpointing instead saves any subset of these activations during the forward pass, and recomputes the intermediate ones just-in-time during the backward pass. Gradient checkpointing can reduce the training memory footprint of a $l$ layer network to $O(\sqrt{l})$.

How does this work? Suppose we have a $l$ layer network divided in $d$ partitions. Only activations at the partition boundaries are saved and communicated between workers. Intermediate activations in intra-partition layers are recomputed during the backward pass. The memory cost for training is thus:

$$
\begin{align}
M(\ell) =\max_{i=1,\dots,k} \underbrace{\text{cost-of-one-partition}(i)}_\text{cost of back-propagation on the i-th partition} + \underbrace{O(d)}_\text{store intermediate outputs} = O(\frac{\ell}{d}) + O(d)
\end{align}
$$

The minimum memory cost is therefore $O(\sqrt{l})$ at $d = \sqrt{l}$.

#### References

[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
[Activation Recomputation](https://lilianweng.github.io/posts/2021-09-25-train-large/#activation-recomputation)

### Graph Compilation

PyTorch operates by default in eager execution mode, whereby operations are executed immediately as they are encountered in Python code. Graph mode is a different method of execution, whereby a computational graph is constructed representing the model's operations. This graph is then optimized an executed as a whole. This can lead to better performance because kernels can be fused into a single kernel, and the graph as a whole can be optimized and pruned for best performance. In practice, graph compilation in PyTorch can yield noticeable speedups, but it can also lead to unintended performance degradations and unnecessary or unexpected graph recompilations. So it's best to experiment with this right before launching any large scale training jobs, as compiling during daily research can yield negligible speedups and add unnecessary breakages and behavior.

Pros:
    - *Faster Execution*: Operations and kernels can be fused, and the graph as a whole can be optimized.

Cons:
    - *Difficult to Debug*: Execution is no longer line-by-line, so operations are difficult to debug.

#### References

[Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
[torch.compile, the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit?tab=t.0#heading=h.ivdr7fmrbeab)

### Data Parallelism

Data Parallelism is a way to train on larger batches of data in parallel, increasing the throughput of the training process.  Typically, Data Parallelism involves copying the model weights into multiple workers, and assigns a fraction of the data to each worker to be processed at the same time. This works well when the model size fits into a single GPU node's memory, but needs to be augmented with additional techniques if the model is too large to fit into a single node's GPU memory.

The key technical challenge in Data Parallel workloads is ensuring the gradients and weights of the model get synchronized across each worker. Typically, at the end of each minibatch, workers need to synchronize gradients or weights to avoid staleness. There are two main approaches to data synchronization:

**Bulk Synchronous Parallel (BSP)**: Workers sync data at the end of every minibatch. It prevents model weights staleness and good learning efficiency but each machine has to halt and wait for others to send gradients. Communication overhead can be large, and so a fast networking interconnect is necessary.

**Asynchronous Parallel (ASP)**: Every GPU worker processes the data asynchronously, no waiting or stalling. This can easily lead to stale weights being used and thus lower the statistical learning efficiency. Even though it increases the computation time, it may not speed up training time to convergence.

A middle ground approach is to synchronize the gradients globally once every N iterations, similar to Gradient Accumulation that we discussed above. PyTorch's [Distributed Data Parallel](https://pytorch.org/docs/main/notes/ddp.html) implementation uses this approach to skip bulk synchronous gradient synchronization. In addition, they use two additional techniques - bucketing gradients and overlapping computation with communication - is used to achieve even better throughput.

The following is the pseudocode for PyTorch's Distributed Data Parallel implementation, to give you an idea of its operation:

![DDP PseudoCode](https://lilianweng.github.io/posts/2021-09-25-train-large/pytorch-ddp.png)

#### References

[PyTorch Data Parallel Best Practices](https://medium.com/pytorch/pytorch-data-parallel-best-practices-on-google-cloud-6c8da2be180d)
[DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
[Distributed Data Parallel](https://pytorch.org/docs/main/notes/ddp.html)
[Measuring the Effects of Data Parallelism on Neural Network Training](https://www.jmlr.org/papers/volume20/18-789/18-789.pdf)
[Data Parallelism](https://lilianweng.github.io/posts/2021-09-25-train-large/#data-parallelism)
[PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704)

### Model Parallelism

Model parallelism is a technique to use when the model weights cannot fit into the GPU memory of single node. Model weights and computation are sharded across multiple workers. In contrast to data parallelism where each worker hosts a full copy of the model, model parallelism only allocates a fraction of the model parameters to each worker, thereby reducing the memory requirements and computation on each worker.

Since deep neural networks usually contain a stack of vertical layers (and modern transformer architectures more so), it feels natural to partition the models layers into $d$ partitions, where a small consecutive set of layers are grouped into the partion on one worker. However, a naive implementation with sequential dependencies leads to big bubbles of waiting time and severe under-utilization of GPU resources.

![Model Parallel](https://lilianweng.github.io/posts/2021-09-25-train-large/naive-data-parallelism.png)

So while you can effectively train a large model using this technique, there is a lot of room for improvement in resource utilization.

#### References

[Single-Machine Model Parallel Best Practices](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
[Model Parallelism](https://lilianweng.github.io/posts/2021-09-25-train-large/#model-parallelism)

### Pipeline Parallelism

Training large models can lead to out-of-memory when the size of the model is too large for a single GPU. To train such a large model, layers can be pipelined across different GPU devices. This is the basic idea in model parallelism. Pipeline parallelism extends this by pipelining the data and computation on each worker for more efficient resource utilization.
The main idea is to split each minibatch of data into $m$ microbatches, and let each partition worker process one microbatch simultaneously. How these microbatches are processed, and how and when the gradients are aggregated, is the subject of a number of different approaches. Note each microbatch involves 1 forward and 1 backward pass.

[GPipe](https://arxiv.org/pdf/1811.06965) aggregates and synchronizes the microbatch gradients at the end of each batch. Synchronously processing the gradients after all of the microbatches have been processed guarantees learning consistency and ensures no workers are operating with stale weights. Bubbles of idle time still exist, but they can be managed effectively, and in the paper, the bubble overhead is almost negligible when $ m > 4d$ where $m$ is the number of microbatches and $d$ is the number of partitions.

![GPipe Pipeline Parallelism](https://lilianweng.github.io/posts/2021-09-25-train-large/gpipe.png)

GPipe achieves almost linear speedup in throughput with the number of devices, although it is not always guaranteed if the model parameters are not evenly distributed across workers.

[PipeDream](https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_pipedream.pdf) is another approach that schedules each worker to alternatively process one forward and one backward pass for each microbatch. PipeDream uses a round robin scheduling strategy to ensure that each worker is fully utilized and ensuring that the forward and backward pass for each minibatch are processed by the same replica of worker stages.

![PipeDream Pipeline Parallelism](https://lilianweng.github.io/posts/2021-09-25-train-large/pipedream.png)

PipeDream does not have a global end of batch synchronization phase, so a naive implementation could easily lead to the forward and backward passes of each microbatch on one worker using different model weights. PipeDream proposed two techniques to alleviate this:

- *Weight Stashing*: Each worker keeps multiple versions of the model weights and ensures that the forward and backward passes for each microbatch use the same version of the model weights.
- *Vertical Sync (optional)*: The version of model weights flows between partition workers along with the activations and the gradients. The forward and backward passes then use the stashed version of the model weights propagated from the previous worker.

Two further improvements to PipeDream - PipeDream-Flush and PipeDream-2BW - were proposed in the paper [Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2006.09503).

PipeDream-Flush adds a globally synchronized pipeline flush periodically, similar to GPipe. This saves a lot of memory as you don't need to store all of the intermediate model variations, at a small cost reduced efficiency (idle time bubble).
![PipeDream-Flush](https://lilianweng.github.io/posts/2021-09-25-train-large/pipedream-flush.png)

PipeDream-2BW only stores two versions of the models weights, hence "2BW" stands for double buffered. It generates a new model version every $k$ microbatches, where $k > d$ the pipeline depth. Since only two model versions are stored, the memory requirements are significantly reduced over PipeDream.

![Pipeline-2BW](https://lilianweng.github.io/posts/2021-09-25-train-large/pipedream-2bw.png)

#### References

[Pipeline Parallelism](https://pytorch.org/docs/stable/distributed.pipelining.html)
[FairScale Pipeline Parallelism](https://fairscale.readthedocs.io/en/latest/deep_dive/pipeline_parallelism.html)
[Tutorial](https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html)
[Pipeline Parallelism](https://lilianweng.github.io/posts/2021-09-25-train-large/#pipeline-parallelism)
[DeepSpeed Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
[FairScale Pipeline Parallelism](https://fairscale.readthedocs.io/en/latest/deep_dive/pipeline_parallelism.html)
[PipeDream Github - MSR Fiddle](https://github.com/msr-fiddle/pipedream)

#### Papers
[GPipe](https://arxiv.org/pdf/1811.06965)
[TorchGPipe](https://arxiv.org/pdf/2004.09910)
[PipeDream: Generalized Pipeline Parallelism for DNN Training](https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_pipedream.pdf)
[Memory-Efficient Pipeline-Parallel DNN Training](https://arxiv.org/abs/2006.09503)
[BPipe: Memory-Balanced Pipeline Parallelism for Training Large Language Models](https://proceedings.mlr.press/v202/kim23l/kim23l.pdf)
[Merak: An Efficient Distributed DNN Training Framework With Automated 3D Parallelism for Giant Foundation Models](https://arxiv.org/abs/2206.04959)
[AutoPipe: A Fast Pipeline Parallelism Approach with Balanced Partitioning and Micro-batch Slicing](https://ieeexplore.ieee.org/document/9912711)
[PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models](https://proceedings.mlr.press/v139/he21a.html)
[Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)

### Tensor Parallelism

Tensor Parallelism was originally proposed in the [Megatron-LM](https://arxiv.org/abs/1909.08053) paper as an additional technique to scale large transformer model training. It involves splitting a tensor operation row or column-wise, and sending the sharded operation to different workers. For example, in a standard MLP, you multiply a tensor `X` by a weights tensor `Y`. Since the tensor dot product is a column-wise operation on `Y`, you can split the columns of `Y = [Y_1, ...., Y_N]` across `N` GPUs, perform each column-wise operation independently, and gather the results at the end. Megatron-LM introduced this approach to shard both the MLP and the self-attention layers of a transformer

![Megatron-LM TP](https://lilianweng.github.io/posts/2021-09-25-train-large/Megatron-LM.png)

#### References

[Large Scale Transformer model training with Tensor Parallel (TP)](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
[MegatronLM: Training Billion+ Parameter Language Models Using GPU Model Parallelism](https://nv-adlr.github.io/MegatronLM)
[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
[TAP: Accelerating Large-Scale DNN Training Through Tensor Automatic Parallelisation](https://arxiv.org/abs/2302.00247)
[Tensor Parallelism](https://lilianweng.github.io/posts/2021-09-25-train-large/#tensor-parallelism)

### Sequence Parallelism

Sequence Parellelism is a technique to parallelize the input to the model, and was introduced in the paper [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198), along with activation recomputation. For example, in transformer based models, the LayerNorm and Dropout layers after the MLP are independent along the sequence dimension. In combination with Tensor Parallelism, sequence parallelism we can more efficiently shard operation with high activation memory usage across GPUs.

![Sequence Parallelism](https://drive.google.com/uc?export=view&id=1D7KEnVDET8uNL050e9jGjxERD2miBwoi)

#### References

[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
[Speed Up by Pipelining Inputs](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs)

### Mixture of Experts

With the Mixture-of-Experts (MoE)⁠ approach, only a fraction of the network is used to compute the output for any one input. One example approach is to have many sets of weights and the network can choose which set to use via a gating mechanism at inference time. This enables many more parameters without increased computation cost. Each set of weights is referred to as “experts,” in the hope that the network will learn to assign specialized computation and skills to each expert. Different experts can be hosted on different GPUs, providing a clear way to scale up the number of GPUs used for a model.

#### References

[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
[Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
[Mixture of Experts](https://lilianweng.github.io/posts/2021-09-25-train-large/#mixture-of-experts-moe)
[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

### CPU Offloading

CPU offloading is a simple approach to tradeoff GPU memory for CPU memory and latency.
CPU offloading temporarily offloads unused data to the CPU or amongst different devices and later reads it back when needed. Naive implementations will slow down training a lot, but sophisticated implementations will pre-fetch data so that the device never needs to wait on it. One implementation of this idea is ZeRO⁠, from the paper [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), which splits the parameters, gradients, and optimizer states across all available hardware and materializes them as needed.

#### References

[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
[GeePS: Scalable deep learning on distributed GPUs with a GPU-specialized parameter server](https://www.pdl.cmu.edu/PDL-FTP/CloudComputing/GeePS-cui-eurosys16.pdf)
[ZeRO Overview](https://www.deepspeed.ai/tutorials/zero/#zero-overview)

### Memory Efficient Optimizers

Memory Efficient Optimizers have been proposed to reduce the memory footprint of the running state maintained by the optimizer. Once such example is [Adafactor⁠](https://arxiv.org/abs/1804.04235).

#### References

[Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)
[Memory-Efficient Adaptive Optimization](https://arxiv.org/abs/1901.11150)
[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

### Training Compression

Compression also can be used for storing intermediate results in the network. For example, [Gist⁠](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/fiddle-gist-isca18.pdf) compresses activations that are saved for the backward pass. DALL·E⁠ compresses the gradients before synchronizing them, according to Lilian Weng from OpenAI. Since the training code was never released we cannot verify this independently or examine the compression implementation.

#### References

[Gist: Efficient Data Encoding for Deep Neural Network Training](https://www.microsoft.com/en-us/research/uploads/prod/2018/04/fiddle-gist-isca18.pdf)

### Fused Kernels

Another popular approach to speeding up training for large models is writing custom kernels with fused operations, to minimize the exchange of data between RAM and VRAM. Take for example a naive, pure PyTorch implmentation of softmax. The code looks like:

```
def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret
```

When implemented naively in PyTorch, computing `y = naive_softmax(x)` for $x \in R^{MxN}$ requires reading $5MN + 2M$ elements from DRAM and writing back $3MN + 2M$ elements. This is absolutely very wasteful, as you should only need to read/write `x` ($MN$ elements) once from DRAM and back. So "kernel fusion" aims to eliminate this bottleneck, by "fusing" all of the operations into a single, larger operation.

[Triton](https://github.com/triton-lang/triton) is a popular framework developed by OpenAI for writing CUDA compatible kernels in a Python-like programming language.

#### References

[Introducing Triton: Open-source GPU programming for neural networks](https://openai.com/index/triton/)
[Kernel Fusion: An Effective Method for Better Power Efficiency on Multithreaded GPU](https://ieeexplore.ieee.org/document/5724850)

### Network Bandwidth

A large part of large scale training is managing network communication between intra- and inter-node workers.

#### Connection Backends

[GLOO](https://github.com/facebookincubator/gloo)
[NCCL](https://github.com/nvidia/nccl)
[NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#collective-operations)
[MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)
[iPerf3](https://github.com/esnet/iperf)

## Training

![Training Flowchart](https://fairscale.readthedocs.io/en/stable/_images/flowchart.png)

Eleuther uses 3D-Parallel ZeRO-1 with pipeline parallelism across nodes and tensor parallelism within nodes.

Megatron-LM ([Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)]) combined pipeline, tensor, and data parallelism, along with a new pipeline scheduling algorithm (interleaved 1F1B), into an approach they called PTD-P.

### Training Infrastructure

[Terraform](https://www.terraform.io/)
[SLURM](https://slurm.schedmd.com/documentation.html)
[NVLink](https://www.nvidia.com/en-us/data-center/nvlink/)
[NCCL Fast Socket](https://github.com/google/nccl-fastsocket)
[NCCL](https://developer.nvidia.com/nccl)
[Kubeflow Pytorch Operators]()
[Volcano Pytorch Operators]()
[SkyPilot]()

### Training Frameworks

[DeepSpeed](https://github.com/microsoft/DeepSpeed)
[Accelerate](https://github.com/huggingface/accelerate)
[Megatron-LM](https://huggingface.co/docs/accelerate/en/usage_guides/megatron_lm)
[Colossal AI](https://github.com/hpcaitech/ColossalAI)
[Trading compute for memory in PyTorch models using Checkpointing](https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb)
[Gradient Checkpointing](https://medium.com/geekculture/training-larger-models-over-your-average-gpu-with-gradient-checkpointing-in-pytorch-571b4b5c2068)
[Gradient Checkpointing in Tensorflow](https://github.com/cybertronai/gradient-checkpointing)
[Gradient Checkpointing in PyTorch](https://pytorch.org/docs/stable/checkpoint.html)
[Fair Scale](https://fairscale.readthedocs.io/en/stable/what_is_fairscale.html)
[Techniques for Training Really Large Neural Networks](https://openai.com/index/techniques-for-training-large-neural-networks/)
[GPT-Neox](https://github.com/EleutherAI/gpt-neox)

### DeepSpeed

DeepSpeed is a lightweight Pytorch training and inference optimization library designed to make using large models easy. For training, supports training models up to 13b parameters on single GPU, without using model parallelism techniques, which can be hard to implement. DeepSpeed reduces the training memory footprint through a novel solution called Zero Redundancy Optimizer (ZeRO). Unlike basic data parallelism where memory states are replicated across data-parallel processes, ZeRO partitions model states and gradients to save significant memory. Furthermore, it also reduces activation memory and fragmented memory. DeepSpeed includes a range of other optimization techniques for training large models, including:

- Mixed Precision Training
- 3D Parallelism
    - DDP + Tensor Parallelism + Pipeline Parallelism
- Zero Redundancy Optimizer
- CPU offloading
- Dense transformer kernels
- Sparse Attention
- Low-bit optimizers
    - 1-bit Adam
    - 0/1 Adam
    - 1-bit LAMB
- Smart Gradient Accumulation
- Overlapping communication/computation
- Curriculum learning
- Progressive layer dropping
- Mixture of Experts

#### References

[DeepSpeed: Extreme-scale model training for everyone](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)
[DeepSpeed Training](https://www.deepspeed.ai/training/)
[DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
[ZeRO-1](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
[ZeRO-2](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/)
[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)
[ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209)

### Megatron

Megatron from NVIDIA is both a distributed training framework as well as a series of language models built using that training framework. Megatron-LM was the first language model to hit 1T parameters, for example. In the training framework context, Megatron consists of two parts: Megatron-LM and Megatron-Core.

Megatron-LM serves as a research-oriented framework leveraging Megatron-Core for large language model (LLM) training

Megatron-Core, on the other hand, is a library of GPU optimized training techniques that comes with formal product support including versioned APIs and regular releases. Megatron-Core offers core building blocks such as attention mechanisms, transformer blocks and layers, normalization layers, and embedding techniques. Additional functionality like activation recomputation, distributed checkpointing is also natively built-in to the library. The building blocks and functionality are all GPU optimized, and can be built with advanced parallelization strategies for optimal training speed and stability. Another key component of the Megatron-Core library includes advanced model parallelism techniques (tensor, sequence, pipeline, context, and MoE expert parallelism).

#### References

[GitHub](https://github.com/NVIDIA/Megatron-LM)
[Megatron-LM Examples](https://github.com/NVIDIA/Megatron-LM/tree/main/examples)
[Megatron-Core Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)

### FairScale

FairScale is a PyTorch extension library for high performance and large scale training. This library extends basic PyTorch capabilities while adding new SOTA scaling techniques. FairScale makes available the latest distributed training techniques in the form of composable modules and easy to use APIs. These APIs are a fundamental part of a researcher's toolbox as they attempt to scale models with limited resources. FairScale provided the first implementation of FDSP for example, after which it was upstreamed into mainline PyTorch.

FairScale is a more research focused version of large scale training algorithms than is released in mainline PyTorch. As such, it contains more SOTA scaling techniques and makes available the latest distributed training techniques in the form of composable modules and easy to use APIs. FairScale does not see a lot of updates anymore, but it is still useful to dive in and play with the different scaling techniques to get a full picture of they work.

#### References

[GitHub](https://github.com/facebookresearch/fairscale)
[Documentation](https://fairscale.readthedocs.io/en/latest/)

### LLM Foundry

#### References
[Github](https://github.com/mosaicml/llm-foundry)

### GPT-Neox

#### References

[Github](https://github.com/EleutherAI/gpt-neox)

### Megablocks

#### References

[Github](https://github.com/databricks/megablocks)

### Horovod

#### References

[Github](https://github.com/horovod/horovod)

### Lightning Fabric

#### References

[Lightning Fabric](https://lightning.ai/docs/fabric/stable/)

### Apex

#### References

[Github](https://github.com/NVIDIA/apex)
[Documentation](https://nvidia.github.io/apex/)

### Nanotron

Nanotron is a library for pretraining transformer models. It provides a simple and flexible API to pretrain models on custom datasets. Nanotron is designed to be easy to use, fast, and scalable. It is built with the following principles in mind:

- Simplicity: Nanotron is designed to be easy to use. It provides a simple and flexible API to pretrain models on custom datasets.
- Performance: Optimized for speed and scalability, Nanotron uses the latest techniques to train models faster and more efficiently.

#### References

[Nanotron](https://github.com/huggingface/nanotron)

### Accelerate

#### References

[Accelerate Examples](https://github.com/huggingface/accelerate/tree/main/examples)

### Torch Primitives

[Writing Distributed Applications with PyTorch](https://pytorch.org/tutorials/intermediate/dist_tuto.html#writing-distributed-applications-with-pytorch)

#### Distributed Data Parallel

##### References

[PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704)
[Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

#### Fully Sharded Data Parallel

PyTorch FDSP grew out of FairScale's research on large model training.  FDSP shards an AI model’s parameters across data parallel workers and can optionally offload part of the training computation to the CPUs. As its name suggests, FSDP is a type of data-parallel training algorithm. Although the parameters are sharded to different GPUs, the computation for each microbatch of data is still local to each GPU worker. It improves memory efficiency by sharding model parameters, gradients, and optimizer states across GPUs, and improves computational efficiency by decomposing the communication and overlapping it with both the forward and backward passes.

In standard data parallel training methods, a copy of the model is present on each GPU and a sequence of forward and backward passes are evaluated on only a shard of the data. After these local computations, the parameters and optimizers for each local process are shared with the other GPUs in order to calculate the global weight update. In FSDP, only a shard of the model is present on a GPU. Then, locally, all weights are gathered from the other GPUs — by means of an all-gather step — to calculate the forward pass. This gathering of weights is then performed again before the backward pass. After that backward pass, the local gradients are averaged and sharded across the GPUs by means of a reduce-scatter step, which allows each GPU to update its local weight shard.

FDSP provides a lot of the same functionality, albeit with different implementations, as DeepSpeed. Full sharding (`FULL_SHARD`) in Pytorch FDSP corresponds to DeepSpeed ZeRO-3, which shards optimizer states, gradients, and model parameters. Gradient sharding (`SHARD_GRAD_OP`) corresponds to ZeRO-2, which shards optimizer states and gradients. Hybrid sharding (`HYBID_SHARD`) corresponds to ZeRO++ Stage-3. This will shard optimizer states, gradients and model parameters within each node while each node has full copy. 

![FDSP Workflow](https://pytorch.org/assets/images/fsdp_workflow.png)

##### References

[PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
[FairScale FDSP](https://engineering.fb.com/2021/07/15/open-source/fsdp/)
[Introducing PyTorch Fully Sharded Data Parallel (FSDP) API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
[PyTorch](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
[FDSP vs DeepSpeed](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed)
[Fully Sharded Data Parallel: faster AI training with fewer GPUs](https://engineering.fb.com/2021/07/15/open-source/fsdp/)
[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
[Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training](https://arxiv.org/abs/2004.13336)
[Maximizing Training Throughput Using PyTorch FSDP and Torch.compile](https://pytorch.org/blog/maximizing-training-throughput/)



## Papers
[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
[GeePS: Scalable deep learning on distributed GPUs with a GPU-specialized parameter server](https://www.pdl.cmu.edu/PDL-FTP/CloudComputing/GeePS-cui-eurosys16.pdf): Early DDP with offloading unused parameters to CPU
[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
[PipeDream: Generalized Pipeline Parallelism for DNN Training](https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_pipedream.pdf)
[Aceso: Efficient Parallel DNN Training through Iterative Bottleneck Alleviation](https://dl.acm.org/doi/pdf/10.1145/3627703.3629554)

## Github
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
[vLLM](https://github.com/vllm-project/vllm)
[GPT-Neox](https://github.com/EleutherAI/gpt-neox)
[trl](https://github.com/huggingface/trl)
[Horovod](https://github.com/horovod/horovod)


## Debugging Tips and Tricks

```
export LOGLEVEL="DEBUG"
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export ACCELERATE_DEBUG_MODE="1"
export NCCL_DEBUG=INFO 
```

## Not Yet Categorized
[1] Saving memory using gradient-checkpointing: https://github.com/cybertronai/gradient-checkpointing

[2] Fitting larger networks into memory: https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9

[4] Deep Learning Memory Usage and Pytorch Optimization Tricks: https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks

Gradient Checkpointing: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
Gradient Checkpointint 2: [Divide-and-Conquer Checkpointing for Arbitrary Programs with No User Annotation](https://arxiv.org/abs/1708.06799)

https://medium.com/pytorch/pytorch-data-parallel-best-practices-on-google-cloud-6c8da2be180d

[Zeno](https://zenoml.com/)
[Torch Run](https://pytorch.org/docs/stable/elastic/run.html)
[Accelerate Multi-GPU](https://github.com/huggingface/accelerate/blob/main/examples/README.md#simple-nlp-example)
https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling

[Lightning Fabric](https://lightning.ai/docs/fabric/stable/)
[Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
[Writing Distributed Applications with PyTorch](https://pytorch.org/tutorials/intermediate/dist_tuto.html#writing-distributed-applications-with-pytorch)
[Triton](https://github.com/triton-lang/triton
)
[Project Fiddle](https://www.microsoft.com/en-us/research/project/fiddle/)
[NanoGPT](https://github.com/karpathy/nanoGPT)
[Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt)
[Unsloth](https://unsloth.ai/) ([Github](https://github.com/unslothai/unsloth))
[Volcano](https://volcano.sh/en/)
[SLURM Basics](https://stanford-rc.github.io/docs-earth/docs/slurm-basics)
[MS-AMP](https://github.com/Azure/MS-AMP)
[Transformer Math 101](https://blog.eleuther.ai/transformer-math/)
[Megatron Training Example](https://github.com/facebookresearch/fairseq/blob/main/examples/megatron_11b/README.md)
[Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034)
[Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20](https://github.com/karpathy/llm.c/discussions/481)
