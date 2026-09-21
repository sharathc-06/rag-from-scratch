# Autograd mechanics

Source: https://docs.pytorch.org/docs/stable/notes/autograd.html
(Fetched via versioned mirror docs.pytorch.org/docs/2.14/notes/autograd.html — stable redirects there)
Created On: Jan 16, 2017 | Last Updated On: Aug 03, 2026

> NOTE ON COMPLETENESS: This is the longest page in the concepts category (includes a full derivation of Wirtinger calculus for complex-number autograd). The fetch was truncated mid-sentence at the start of the "Backward Hooks execution" section, right after the "Hooks for saved tensors" section completed. Everything above that point is a verbatim capture from a single full-page fetch. The "Backward Hooks execution" section and anything after it on the live page is NOT included here and was not reconstructed.

This note will present an overview of how autograd works and records the operations. It's not strictly necessary to understand all this, but we recommend getting familiar with it, as it will help you write more efficient, cleaner programs, and can aid you in debugging.

## How autograd encodes the history

Autograd is a reverse automatic differentiation system. Conceptually, autograd records a graph recording all of the operations that created the data as you execute operations, giving you a directed acyclic graph whose leaves are the input tensors and roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.

Internally, autograd represents this graph as a graph of `Function` objects (really expressions), which can be `apply()`ed to compute the result of evaluating the graph. When computing the forward pass, autograd simultaneously performs the requested computations and builds up a graph representing the function that computes the gradient (the `.grad_fn` attribute of each `torch.Tensor` is an entry point into this graph). When the forward pass is completed, we evaluate this graph in the backwards pass to compute the gradients.

An important thing to note is that the graph is recreated from scratch at every iteration, and this is exactly what allows for using arbitrary Python control flow statements, that can change the overall shape and size of the graph at every iteration. You don't have to encode all possible paths before you launch the training - what you run is what you differentiate.

### Saved tensors

Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass. For example, the function x -> x^2 saves the input x to compute the gradient.

When defining a custom Python `Function`, you can use `save_for_backward()` to save tensors during the forward pass and `saved_tensors` to retrieve them during the backward pass. See "Extending PyTorch" for more information.

For operations that PyTorch defines (e.g. `torch.pow()`), tensors are automatically saved as needed. You can explore (for educational or debugging purposes) which tensors are saved by a certain `grad_fn` by looking for its attributes starting with the prefix `_saved`.

```python
x = torch.randn(5, requires_grad=True)
y = x.pow(2)
print(x.equal(y.grad_fn._saved_self))  # True
print(x is y.grad_fn._saved_self)  # True
```

In the previous code, `y.grad_fn._saved_self` refers to the same Tensor object as `x`. But that may not always be the case. For instance:

```python
x = torch.randn(5, requires_grad=True)
y = x.exp()
print(y.equal(y.grad_fn._saved_result))  # True
print(y is y.grad_fn._saved_result)  # False
```

Under the hood, to prevent reference cycles, PyTorch has *packed* the tensor upon saving and *unpacked* it into a different tensor for reading. Here, the tensor you get from accessing `y.grad_fn._saved_result` is a different tensor object than `y` (but they still share the same storage).

Whether a tensor will be packed into a different tensor object depends on whether it is an output of its own `grad_fn`, which is an implementation detail subject to change and that users should not rely on.

You can control how PyTorch does packing / unpacking with Hooks for saved tensors (see below).

## Gradients for non-differentiable functions

The gradient computation using Automatic Differentiation is only valid when each elementary function being used is differentiable. Unfortunately many of the functions we use in practice do not have this property (`relu` or `sqrt` at `0`, for example). To try and reduce the impact of functions that are non-differentiable, we define the gradients of the elementary operations by applying the following rules in order:

1. If the function is differentiable and thus a gradient exists at the current point, use it.
2. If the function is convex (at least locally), use a subgradient of minimum norm.
3. If the function is concave (at least locally), use a supergradient of minimum norm (consider `-f(x)` and apply the previous point).
4. If the function is defined, define the gradient at the current point by continuity (note that `inf` is possible here, for example for `sqrt(0)`). If multiple values are possible, pick one arbitrarily.
5. If the function is not defined (`sqrt(-1)`, `log(-1)` or most functions when the input is `NaN`, for example) then the value used as the gradient is arbitrary (we might also raise an error but that is not guaranteed). Most functions will use `NaN` as the gradient, but for performance reasons, some functions will use other values (`log(-1)`, for example).
6. If the function is not a deterministic mapping (i.e. it is not a mathematical function), it will be marked as non-differentiable. This will make it error out in the backward if used on tensors that require grad outside of a `no_grad` environment.

In particular, a `NaN` input is outside the mathematical input domain used by these rules, even when the operator has specified floating-point behavior for `NaN`. An autograd formula may return any gradient at such an input and does not need to propagate the `NaN` input into the gradient. This is distinct from a `NaN` incoming gradient or tangent: formulas must propagate it when that contribution is used, without allowing an inactive contribution to contaminate the result.

For these rules, the function and its input space are defined by the selected dispatcher overload and its schema signature. This applies to built-in operators and user-defined operators, including Python `custom_op` definitions. Overload resolution occurs before autograd applies these rules; Python-level syntax does not redefine the function's input space.

All differentiable Tensor arguments in the selected dispatcher signature jointly form the function's input space, regardless of whether a particular argument currently requires grad. `requires_grad` only controls which gradient components autograd computes. Non-Tensor arguments, including Scalar parameters, are fixed parameters. The norm is the Euclidean norm on this joint Tensor input space.

Consequently, different dispatcher signatures can intentionally produce different subgradients. Consider these two schemas:

```
max(Tensor x, Scalar c) -> Tensor
max.Tensor(Tensor x, Tensor y) -> Tensor
```

At equality, the first schema treats `c` as fixed, so the minimum-norm derivative with respect to `x` is `0`. The second schema is jointly a function of `x` and `y`, so its minimum-norm joint subgradient is `(1/2, 1/2)`. A zero-dimensional Tensor remains a Tensor argument for this purpose.

### Division by Zero in Autograd

When performing division by zero in PyTorch (e.g., `x / 0`), the forward pass will produce `inf` values following IEEE-754 floating point arithmetic. While these `inf` values can be masked out before computing the final loss (e.g., via indexing or masking), the autograd system still tracks and differentiates through the full computation graph, including the division by zero operation.

During backpropagation, this can lead to problematic gradient expressions. For example:

```python
x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])

y = x / div          # Results in [inf, 1]
mask = div != 0      # [False, True]
loss = y[mask].sum()
loss.backward()
print(x.grad)        # [nan, 1], not [0, 1]
```

In this example, even though we only use the masked output (which excludes the division by zero), autograd still computes gradients through the full computation graph, including the division by zero operation. This results in `nan` gradients for the masked elements, which can cause training instability.

To avoid this issue, there are several recommended approaches:

1. Mask before division:

```python
x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])

mask = div != 0
safe = torch.zeros_like(x)
safe[mask] = x[mask] / div[mask]
loss = safe.sum()
loss.backward()      # Produces safe gradients [0, 1]
```

2. Use MaskedTensor (experimental API):

```python
from torch.masked import as_masked_tensor

x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])

y = x / div
mask = div != 0
loss = as_masked_tensor(y, mask).sum()
loss.backward()      # Cleanly handles "undefined" vs "zero" gradients
```

The key principle is to prevent the division by zero operation from being recorded in the computation graph, rather than masking its results after the fact. This ensures that autograd only computes gradients through valid operations.

This behavior is important to keep in mind when working with operations that might produce `inf` or `nan` values, as masking the outputs does not prevent the problematic gradients from being computed.

## Locally disabling gradient computation

There are several mechanisms available from Python to locally disable gradient computation:

To disable gradients across entire blocks of code, there are context managers like no-grad mode and inference mode. For more fine-grained exclusion of subgraphs from gradient computation, there is setting the `requires_grad` field of a tensor.

Below, in addition to discussing the mechanisms above, we also describe evaluation mode (`nn.Module.eval()`), a method that is not used to disable gradient computation but, because of its name, is often mixed up with the three.

### Setting requires_grad

`requires_grad` is a flag, defaulting to false *unless wrapped in a* `nn.Parameter`, that allows for fine-grained exclusion of subgraphs from gradient computation. It takes effect in both the forward and backward passes:

During the forward pass, an operation is only recorded in the backward graph if at least one of its input tensors require grad. During the backward pass (`.backward()`), only leaf tensors with `requires_grad=True` will have gradients accumulated into their `.grad` fields.

It is important to note that even though every tensor has this flag, *setting* it only makes sense for leaf tensors (tensors that do not have a `grad_fn`, e.g., a `nn.Module`'s parameters). Non-leaf tensors (tensors that do have `grad_fn`) are tensors that have a backward graph associated with them. Thus their gradients will be needed as an intermediary result to compute the gradient for a leaf tensor that requires grad. From this definition, it is clear that all non-leaf tensors will automatically have `require_grad=True`.

Setting `requires_grad` should be the main way you control which parts of the model are part of the gradient computation, for example, if you need to freeze parts of your pretrained model during model fine-tuning.

To freeze parts of your model, simply apply `.requires_grad_(False)` to the parameters that you don't want updated. And as described above, since computations that use these parameters as inputs would not be recorded in the forward pass, they won't have their `.grad` fields updated in the backward pass because they won't be part of the backward graph in the first place, as desired.

Because this is such a common pattern, `requires_grad` can also be set at the module level with `nn.Module.requires_grad_()`. When applied to a module, `.requires_grad_()` takes effect on all of the module's parameters (which have `requires_grad=True` by default).

### Grad Modes

Apart from setting `requires_grad` there are also three grad modes that can be selected from Python that can affect how computations in PyTorch are processed by autograd internally: default mode (grad mode), no-grad mode, and inference mode, all of which can be toggleable via context managers and decorators.

| Mode | Excludes operations from being recorded in backward graph | Skips additional autograd tracking overhead | Tensors created while the mode is enabled can be used in grad-mode later | Examples |
|---|---|---|---|---|
| default | | | ✓ | Forward pass |
| no-grad | ✓ | | ✓ | Optimizer updates |
| inference | ✓ | ✓ | | Data processing, model evaluation |

### Default Mode (Grad Mode)

The "default mode" is the mode we are implicitly in when no other modes like no-grad and inference mode are enabled. To be contrasted with "no-grad mode" the default mode is also sometimes called "grad mode".

The most important thing to know about the default mode is that it is the only mode in which `requires_grad` takes effect. `requires_grad` is always overridden to be `False` in both the two other modes.

### No-grad Mode

Computations in no-grad mode behave as if none of the inputs require grad. In other words, computations in no-grad mode are never recorded in the backward graph even if there are inputs that have `require_grad=True`.

Enable no-grad mode when you need to perform operations that should not be recorded by autograd, but you'd still like to use the outputs of these computations in grad mode later. This context manager makes it convenient to disable gradients for a block of code or function without having to temporarily set tensors to have `requires_grad=False`, and then back to `True`.

For example, no-grad mode might be useful when writing an optimizer: when performing the training update you'd like to update parameters in-place without the update being recorded by autograd. You also intend to use the updated parameters for computations in grad mode in the next forward pass.

The implementations in torch.nn.init also rely on no-grad mode when initializing the parameters as to avoid autograd tracking when updating the initialized parameters in-place.

### Inference Mode

Inference mode is the extreme version of no-grad mode. Just like in no-grad mode, computations in inference mode are not recorded in the backward graph, but enabling inference mode will allow PyTorch to speed up your model even more. This better runtime comes with a drawback: tensors created in inference mode will not be able to be used in computations to be recorded by autograd after exiting inference mode.

Enable inference mode when you are performing computations that do not have interactions with autograd, AND you don't plan on using the tensors created in inference mode in any computation that is to be recorded by autograd later.

It is recommended that you try out inference mode in the parts of your code that do not require autograd tracking (e.g., data processing and model evaluation). If it works out of the box for your use case it's a free performance win. If you run into errors after enabling inference mode, check that you are not using tensors created in inference mode in computations that are recorded by autograd after exiting inference mode. If you cannot avoid such use in your case, you can always switch back to no-grad mode.

### Evaluation Mode (nn.Module.eval())

Evaluation mode is not a mechanism to locally disable gradient computation. It is included here anyway because it is sometimes confused to be such a mechanism.

Functionally, `module.eval()` (or equivalently `module.train(False)`) are completely orthogonal to no-grad mode and inference mode. How `model.eval()` affects your model depends entirely on the specific modules used in your model and whether they define any training-mode specific behavior.

You are responsible for calling `model.eval()` and `model.train()` if your model relies on modules such as `torch.nn.Dropout` and `torch.nn.BatchNorm2d` that may behave differently depending on training mode, for example, to avoid updating your BatchNorm running statistics on validation data.

It is recommended that you always use `model.train()` when training and `model.eval()` when evaluating your model (validation/testing) even if you aren't sure your model has training-mode specific behavior, because a module you are using might be updated to behave differently in training and eval modes.

## In-place operations with autograd

Supporting in-place operations in autograd is a hard matter, and we discourage their use in most cases. Autograd's aggressive buffer freeing and reuse makes it very efficient and there are very few occasions when in-place operations lower memory usage by any significant amount. Unless you're operating under heavy memory pressure, you might never need to use them.

There are two main reasons that limit the applicability of in-place operations:

1. In-place operations can potentially overwrite values required to compute gradients.
2. Every in-place operation requires the implementation to rewrite the computational graph. Out-of-place versions simply allocate new objects and keep references to the old graph, while in-place operations, require changing the creator of all inputs to the `Function` representing this operation. This can be tricky, especially if there are many Tensors that reference the same storage (e.g. created by indexing or transposing), and in-place functions will raise an error if the storage of modified inputs is referenced by any other `Tensor`.

### In-place correctness checks

Every tensor keeps a version counter, that is incremented every time it is marked dirty in any operation. When a Function saves any tensors for backward, a version counter of their containing Tensor is saved as well. Once you access `self.saved_tensors` it is checked, and if it is greater than the saved value an error is raised. This ensures that if you're using in-place functions and not seeing any errors, you can be sure that the computed gradients are correct.

## Multithreaded Autograd

The autograd engine is responsible for running all the backward operations necessary to compute the backward pass. This section will describe all the details that can help you make the best use of it in a multithreaded environment. (This is relevant only for PyTorch 1.6+ as the behavior in previous version was different.)

User could train their model with multithreading code (e.g. Hogwild training), and does not block on the concurrent backward computations, example code could be:

```python
# Define a train function to be used in different threads
def train_fn():
    x = torch.ones(5, 5, requires_grad=True)
    # forward
    y = (x + 3) * (x + 4) * 0.5
    # backward
    y.sum().backward()
    # potential optimizer update

# User write their own threading code to drive the train_fn
threads = []
for _ in range(10):
    p = threading.Thread(target=train_fn, args=())
    p.start()
    threads.append(p)

for p in threads:
    p.join()
```

Note that some behaviors that user should be aware of:

### Concurrency on CPU

When you run `backward()` or `grad()` via python or C++ API in multiple threads on CPU, you are expecting to see extra concurrency instead of serializing all the backward calls in a specific order during execution (behavior before PyTorch 1.6).

### Non-determinism

If you are calling `backward()` from multiple threads concurrently and have shared inputs (i.e. Hogwild CPU training), then non-determinism should be expected. This can occur because parameters are automatically shared across threads, as such, multiple threads may access and try to accumulate the same `.grad` attribute during gradient accumulation. This is technically not safe, and it might result in race condition and the result might be invalid to use.

Users developing multithreaded models featuring shared parameters should have the threading model in mind and should understand the issues described above. The functional API `torch.autograd.grad()` may be used to calculate the gradients instead of `backward()` to avoid non-determinism.

### Graph retaining

If part of the autograd graph is shared between threads, i.e. run first part of forward single thread, then run second part in multiple threads, then the first part of graph is shared. In this case different threads execute `grad()` or `backward()` on the same graph might have issue of destroying the graph on the fly of one thread, and the other thread will crash in this case. Autograd will error out to the user similar to what call `backward()` twice without `retain_graph=True`, and let the user know they should use `retain_graph=True`.

### Thread Safety on Autograd Node

Since Autograd allows the caller thread to drive its backward execution for potential parallelism, it's important that we ensure thread safety on CPU with parallel `backward()` calls that share part/whole of the GraphTask.

Custom Python `autograd.Function`s are automatically thread safe because of GIL. For built-in C++ Autograd Nodes (e.g. AccumulateGrad, CopySlices) and custom `autograd::Function`s, the Autograd Engine uses thread mutex locking to ensure thread safety on autograd Nodes that might have state write/read.

### No thread safety on C++ hooks

Autograd relies on the user to write thread safe C++ hooks. If you want the hook to be correctly applied in multithreading environment, you will need to write proper thread locking code to ensure the hooks are thread safe.

## Autograd for Complex Numbers

The short version:

- When you use PyTorch to differentiate any function f(z) with complex domain and/or codomain, the gradients are computed under the assumption that the function is a part of a larger real-valued loss function g(input)=L. The gradient computed is dL/dz* (note the conjugation of z), the negative of which is precisely the direction of steepest descent used in Gradient Descent algorithm. Thus, there is a viable path in making the existing optimizers work out of the box with complex parameters.
- This convention matches TensorFlow's convention for complex differentiation, but is different from JAX (which computes dL/dz).
- If you have a real-to-real function which internally uses complex operations, the convention here doesn't matter: you will always get the same result that you would have gotten if it had been implemented with only real operations.

### What are complex derivatives?

The mathematical definition of complex-differentiability takes the limit definition of a derivative and generalizes it to operate on complex numbers. Consider a function f: C -> C, f(z=x+yj) = u(x, y) + v(x, y)j, where u and v are two variable real valued functions and j is the imaginary unit.

Using the derivative definition, f'(z) = lim(h->0, h in C) of (f(z+h) - f(z))/h.

In order for this limit to exist, not only must u and v must be real differentiable, but f must also satisfy the Cauchy-Riemann equations. In other words: the limit computed for real and imaginary steps (h) must be equal. This is a more restrictive condition.

The complex differentiable functions are commonly known as holomorphic functions. They are well behaved, have all the nice properties that you've seen from real differentiable functions, but are practically of no use in the optimization world. For optimization problems, only real valued objective functions are used in the research community since complex numbers are not part of any ordered field and so having complex valued loss does not make much sense.

It also turns out that no interesting real-valued objective fulfill the Cauchy-Riemann equations. So the theory with holomorphic function cannot be used for optimization and most people therefore use the Wirtinger calculus.

### Wirtinger Calculus comes into the picture

So, we have this great theory of complex differentiability and holomorphic functions, and we can't use any of it at all, because many of the commonly used functions are not holomorphic. Wirtinger observed that even if f(z) isn't holomorphic, one could rewrite it as a two variable function f(z, z*) which is always holomorphic. This is because real and imaginary of the components of z can be expressed in terms of z and z* as: Re(z) = (z + z*)/2, Im(z) = (z - z*)/(2j).

Wirtinger calculus suggests to study f(z, z*) instead, which is guaranteed to be holomorphic if f was real differentiable. This function has partial derivatives d/dz and d/dz*. Using the chain rule: d/dx = d/dz + d/dz*, and d/dy = 1j * (d/dz - d/dz*).

From these, we get: d/dz = 1/2 * (d/dx - 1j * d/dy), and d/dz* = 1/2 * (d/dx + 1j * d/dy). This is the classic definition of Wirtinger calculus (see Wikipedia).

Consequences: the Cauchy-Riemann equations translate into saying that df/dz* = 0 (the function f can be written entirely in terms of z, without reference to z*). Another important result is that when doing optimization on a real-valued loss, the update step should be given by dLoss/dz* (not dLoss/dz).

### How is Wirtinger Calculus useful in optimization?

Researchers in audio and other fields commonly use gradient descent to optimize real valued loss functions with complex variables, treating the real and imaginary values as separate channels that can be updated. For step size alpha/2 and loss L, in R^2: x_(n+1) = x_n - (alpha/2)*dL/dx, y_(n+1) = y_n - (alpha/2)*dL/dy.

Translating into complex space C: z_(n+1) = z_n - alpha * dL/dz*.

Wirtinger calculus tells us we can simplify the complex variable update formula to only refer to the conjugate Wirtinger derivative dL/dz*, giving exactly the step taken in optimization. Because the conjugate Wirtinger derivative gives exactly the correct step for a real valued loss function, PyTorch gives you this derivative when you differentiate a function with a real valued loss.

### How does PyTorch compute the conjugate Wirtinger derivative?

Derivative formulas take in `grad_output` as an input, representing the incoming Vector-Jacobian product already computed, i.e. dL/ds*, where L is the loss of the entire computation and s is the output of the function. The goal is to compute dL/dz*, where z is the input.

Working through the chain rule with f: C -> C defined as f(z) = f(x+yj) = u(x,y) + v(x,y)j, and assuming f is part of a larger real valued loss function g, the final boxed result is:

dL/dz* = (grad_output)* * ds/dz* + grad_output * (ds/dz)*

This is the important formula for writing your own gradients, decomposing the derivative formula into a simpler one that is easy to compute by hand.

### How can I write my own derivative formula for a complex function?

The boxed equation above gives the general formula for all derivatives on complex functions. You still need ds/dz and ds/dz*. Two ways to do this:
- Use the definition of Wirtinger derivatives directly, calculating ds/dz and ds/dz* using ds/dx and ds/dy.
- Use the change-of-variables trick: rewrite f(z) as a two-variable function f(z, z*), and compute the conjugate Wirtinger derivatives by treating z and z* as independent variables. This is often easier — for a holomorphic function, only z is used and ds/dz* is zero.

Example: f(z = x+yj) = c*z = c*(x+yj), where c is real.

Using the first way: ds/dz = 1/2*(ds/dx - ds/dy*j) = 1/2*(c - (c*1j)*1j) = c. ds/dz* = 1/2*(ds/dx + ds/dy*j) = 1/2*(c + (c*1j)*1j) = 0.

Using grad_output = 1.0 (the default grad output value used when `backward()` is called on a scalar output), dL/dz* = 1*0 + 1*c = c.

Using the second way (treating z, z* independently): ds/dz = d(c*z)/dz = c. ds/dz* = d(c*z)/dz* = 0. Same result, fewer calculations — the second way is generally faster.

### What about cross-domain functions?

Some functions map from complex inputs to real outputs, or vice versa. These form a special case of the boxed formula above:
- For f: C -> R: dL/dz* = 2 * grad_output * ds/dz*
- For f: R -> C: dL/dz* = 2 * Re(grad_output* * ds/dz*)

## Hooks for saved tensors

You can control how saved tensors are packed / unpacked by defining a pair of `pack_hook` / `unpack_hook` hooks. The `pack_hook` function should take a tensor as its single argument but can return any python object (e.g. another tensor, a tuple, or even a string containing a filename). The `unpack_hook` function takes as its single argument the output of `pack_hook` and should return a tensor to be used in the backward pass. The tensor returned by `unpack_hook` only needs to have the same content as the tensor passed as input to `pack_hook`. In particular, any autograd-related metadata can be ignored as they will be overwritten during unpacking.

An example of such a pair is:

```python
class SelfDeletingTempFile():
    def __init__(self):
        self.name = os.path.join(tmp_dir, str(uuid.uuid4()))

    def __del__(self):
        os.remove(self.name)

def pack_hook(tensor):
    temp_file = SelfDeletingTempFile()
    torch.save(tensor, temp_file.name)
    return temp_file

def unpack_hook(temp_file):
    return torch.load(temp_file.name)
```

Notice that the `unpack_hook` should not delete the temporary file because it might be called multiple times: the temporary file should be alive for as long as the returned `SelfDeletingTempFile` object is alive. In the above example, we prevent leaking the temporary file by closing it when it is no longer needed (on deletion of the `SelfDeletingTempFile` object).

Note: We guarantee that `pack_hook` will only be called once but `unpack_hook` can be called as many times as the backward pass requires it and we expect it to return the same data each time.

Warning: Performing inplace operations on the input of any of the functions is forbidden as they may lead to unexpected side-effects. PyTorch will throw an error if the input to a pack hook is modified inplace but does not catch the case where the input to an unpack hook is modified inplace.

### Registering hooks for a saved tensor

You can register a pair of hooks on a saved tensor by calling the `register_hooks()` method on a `SavedTensor` object. Those objects are exposed as attributes of a `grad_fn` and start with the `_raw_saved_` prefix.

```python
x = torch.randn(5, requires_grad=True)
y = x.pow(2)
y.grad_fn._raw_saved_self.register_hooks(pack_hook, unpack_hook)
```

The `pack_hook` method is called as soon as the pair is registered. The `unpack_hook` method is called each time the saved tensor needs to be accessed, either by means of `y.grad_fn._saved_self` or during the backward pass.

Warning: If you maintain a reference to a `SavedTensor` after the saved tensors have been released (i.e. after backward has been called), calling its `register_hooks()` is forbidden. PyTorch will throw an error most of the time but it may fail to do so in some cases and undefined behavior may arise.

### Registering default hooks for saved tensors

Alternatively, you can use the context-manager `saved_tensors_hooks` to register a pair of hooks which will be applied to *all* saved tensors that are created in that context.

Note: With this context manager the input to `pack_hook` is a live tensor that still carries its `grad_fn` (unlike per-tensor `register_hooks()` above, which receives an already-detached tensor). If you keep that live tensor in the object you return and the saved tensor is a graph output, you form a reference cycle: the output's `grad_fn` owns the saved tensor, which then owns a tensor referring back to that same `grad_fn`. This does not produce wrong results. Running backward with `retain_graph=False` releases the saved tensor and breaks the cycle, so memory is freed as usual; but if you build the graph without ever running backward — or use `retain_graph=True` — the cycle keeps the tensor alive, and because PyTorch does not expose the autograd graph to Python's cyclic garbage collector it cannot be reclaimed even by an explicit `gc.collect()`. Calling `.detach()` on the input before keeping it breaks the cycle. Detaching is lossless: PyTorch stashes the autograd metadata separately and restores `grad_fn` on unpack, so gradients are unaffected. Hooks that instead return a freshly computed tensor are unaffected — pack hooks run with gradient tracking disabled, so a tensor computed inside the hook (for example moving a CUDA tensor to CPU) carries no `grad_fn`, and the original input is not retained.

Example:

```python
# Only save on disk tensors that have size >= 1000
SAVE_ON_DISK_THRESHOLD = 1000

def pack_hook(x):
    if x.numel() < SAVE_ON_DISK_THRESHOLD:
        return x.detach()
    temp_file = SelfDeletingTempFile()
    torch.save(tensor, temp_file.name)
    return temp_file

def unpack_hook(tensor_or_sctf):
    if isinstance(tensor_or_sctf, torch.Tensor):
        return tensor_or_sctf
    return torch.load(tensor_or_sctf.name)

class Model(nn.Module):
    def forward(self, x):
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
          # ... compute output
          output = x
        return output

model = Model()
net = nn.DataParallel(model)
```

The hooks defined with this context manager are thread-local. Hence, the following code will not produce the desired effects because the hooks do not go through `DataParallel`.

```python
# Example what NOT to do

net = nn.DataParallel(model)
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    output = net(input)
```

Note that using those hooks disables all the optimization in place to reduce Tensor object creation. For example:

```python
with torch.autograd.graph.saved_tensors_hooks(lambda x: x.detach(), lambda x: x):
    x = torch.randn(5, requires_grad=True)
    y = x * x
```

Without the hooks, `x`, `y.grad_fn._saved_self` and `y.grad_fn._saved_other` all refer to the same tensor object. With the hooks, PyTorch will pack and unpack `x` into two new tensor objects that share the same storage with the original `x` (no copy performed).

<!-- CAPTURE TRUNCATED HERE. The source page continues with a "Backward Hooks execution" section and likely further sections after it, none of which were retrieved in this fetch. Do not treat this file as the complete page. -->
