# Transforms

Source: https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
Created On: Feb 09, 2021 | Last Updated: Aug 11, 2021 | Last Verified: Not Verified

Data does not always come in its final processed form that is required for training machine learning algorithms. We use **transforms** to perform some manipulation of the data and make it suitable for training.

All TorchVision datasets have two parameters - `transform` to modify the features and `target_transform` to modify the labels - that accept callables containing the transformation logic. The `torchvision.transforms` module offers several commonly-used transforms out of the box.

The FashionMNIST features are in PIL Image format, and the labels are integers. For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. To make these transformations, we use the `torchvision.transforms.v2` API along with `torch.nn.functional.one_hot`.

```python
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
)
```

## ToImage / ToDtype

`v2.ToImage` converts a PIL image or NumPy `ndarray` into a `torchvision.tv_tensors.Image` tensor, and `v2.ToDtype` with `scale=True` casts it to float32 and scales the pixel intensity values to the range [0., 1.].

## Lambda Transforms

Lambda transforms apply any user-defined lambda function. Here, we use `torch.nn.functional.one_hot` to turn the integer label into a one-hot encoded tensor of size 10 (the number of labels in our dataset), then cast it to float to match the expected dtype.

```python
target_transform = v2.Lambda(
    lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
)
```

## Further Reading

- torchvision.transforms API

**Total running time of the script:** (0 minutes 4.314 seconds)

---

### Historical note (superseded API)

Earlier versions of this tutorial used the now-deprecated v1 `ToTensor`/`Lambda` pattern with manual `scatter_` for one-hot encoding:

```python
# Deprecated pattern (kept here only for historical/comparison context; do not use in new code)
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

`ToTensor()` converts a PIL image or NumPy `ndarray` into a `FloatTensor` and scales the image's pixel intensity values into the range [0., 1.]. The `scatter_`-based Lambda creates a zero tensor of size 10 and assigns `value=1` at the index given by the label `y`. The current tutorial (above) replaces both of these with the `v2` transforms API and `F.one_hot`.
