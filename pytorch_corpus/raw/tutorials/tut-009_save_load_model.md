# Save and Load the Model

Source: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

In this section we will look at how to persist model state with saving, loading and running model predictions.

```python
import torch
import torchvision.models as models
```

## Saving and Loading Model Weights

PyTorch models store the learned parameters in an internal state dictionary, called `state_dict`. These can be persisted via the `torch.save` method:

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

To load model weights, you need to create an instance of the same model first, and then load the parameters using the `load_state_dict()` method.

In the code below, we set `weights_only=True` to limit the functions executed during unpickling to only those necessary for loading weights. Using `weights_only=True` is considered a best practice when loading weights.

```python
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
```

Note: be sure to call the `model.eval()` method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.

## Saving and Loading Models with Shapes

When loading model weights, we needed to instantiate the model class first, because the class defines the structure of a network. We might want to save the structure of this class together with the model, in which case we can pass `model` (and not `model.state_dict()`) to the saving function:

```python
torch.save(model, 'model.pth')
```

We can then load the model as demonstrated below.

As described in "Saving and loading torch.nn.Modules", saving `state_dict` is considered the best practice. However, below we use `weights_only=False` because this involves loading the model itself, which is a legacy use case for `torch.save`.

```python
model = torch.load('model.pth', weights_only=False)
```

Note: This approach uses Python pickle when serializing the model, thus it relies on the actual class definition to be available when loading the model.

## Related Tutorials

- Saving and Loading a General Checkpoint
- Tips for Loading an nn.Module from a Checkpoint

## Background: state_dict, torch.save, torch.load

PyTorch offers three core functions for model serialization:

- `torch.save`: Saves a serialized object to disk. This function uses Python's pickle utility for serialization. Models, tensors, and dictionaries of all kinds of objects can be saved using this function.
- `torch.load`: Uses pickle's unpickling facilities to deserialize pickled object files to memory. This function also facilitates the device to load the data into.
- `torch.nn.Module.load_state_dict`: Loads a model's parameter dictionary using a deserialized `state_dict`.

In PyTorch, the learnable parameters (i.e. weights and biases) of a `torch.nn.Module` model are contained in the model's parameters (accessed with `model.parameters()`). A `state_dict` is simply a Python dictionary object that maps each layer to its parameter tensor. Note that only layers with learnable parameters (convolutional layers, linear layers, etc.) and registered buffers (batchnorm's `running_mean`) have entries in the model's `state_dict`. Optimizer objects (`torch.optim`) also have a `state_dict`, which contains information about the optimizer's state, as well as the hyperparameters used.

Because `state_dict` objects are Python dictionaries, they can be easily saved, updated, altered, and restored, adding a great deal of modularity to PyTorch models and optimizers.
