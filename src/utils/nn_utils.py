import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
    

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        # self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


net_config = {
    "relu": {
        "class": nn.ReLU, 
        "kwargs": [],
    },
    "avgPool2d": {
        "class": nn.AvgPool2d, 
        "kwargs": ["kernel_size", "stride", "padding"],
    },
    "linear": {
        "class": nn.Linear, 
        "kwargs": ["in_features", "out_features", "bias"],
        "inferFirst": True,
    },
    "batchNorm1d": {
        "class": nn.BatchNorm1d, 
        "kwargs": ["num_features"],
        "inferFirst": True,
    },
    "flatten": {
        "class": nn.Flatten, 
        "kwargs": [],
    },
    "interpolate": {
        "class": Interpolate, 
        "kwargs": ["scale_factor"],
    },
    "GRU": {
        "class": nn.GRUCell,
        "kwargs": ["input_size", "hidden_size"],
        "inferFirst": True,
    },
    "LSTM": {
        "class": nn.LSTMCell,
        "kwargs": ["input_size", "hidden_size"],
        "inferFirst": True,
    }
}

def layer_from_dict(layer_dict, input_shape):
    """
    Build a PyTorch layer from a dictionary representation.
    
    Args:
        layer_dict: Dictionary with 'type' key and optional layer-specific parameters
        input_shape: Shape of the input tensor (tuple)
    
    Returns:
        layer: The constructed PyTorch layer
        output_shape: Shape of the output tensor (tuple)
    """
    assert isinstance(layer_dict, dict), f"Expected dictionary, got {type(layer_dict)}"
    assert "type" in layer_dict, f"Layer dictionary must have 'type' key: {layer_dict}"
    
    layer_type = layer_dict["type"]
    assert layer_type in net_config.keys(), f"Unexpected layer type: {layer_type}"

    kwargs = {}
    layer_config = net_config[layer_type]
    
    # Handle layers that infer first parameter from input_shape
    if layer_config.get("inferFirst", False):
        first_kwarg = layer_config["kwargs"][0]
        if first_kwarg == "shape":
            kwargs[first_kwarg] = input_shape
        else:
            kwargs[first_kwarg] = input_shape[0]
    
    # Add remaining parameters from layer_dict
    for kwarg_name in layer_config["kwargs"]:
        if kwarg_name in layer_dict:
            kwargs[kwarg_name] = layer_dict[kwarg_name]
    
    layer = layer_config["class"](**kwargs)
    
    # Compute output shape
    x = torch.empty(1, *input_shape)
    with torch.no_grad():
        if layer_type.startswith("batchNorm"):
            output_shape = (1, *input_shape)
        elif layer_type == "flatten":
            output_shape = (1, prod(input_shape))
        elif layer_type == "LSTM":
            output_shape = layer(x)[0].shape
        else:
            output_shape = layer(x).shape

    return layer, output_shape[1:]

def net_from_yaml(layer_list, input_shape, target_shape=None):
    """
    Build a PyTorch Sequential network from a list of layer dictionaries.
    
    Args:
        layer_list: List of dictionaries, each with 'type' key and optional parameters
        input_shape: Shape of the input tensor (tuple)
        target_shape: Optional target output shape. If provided and is 1D, a final
                     linear layer will be appended to match this shape.
    
    Returns:
        net: nn.Sequential network
        output_shape: Final output shape (tuple)
    """
    assert isinstance(layer_list, list), f"Expected list of layer dictionaries, got {type(layer_list)}"
    
    # Automatically prepend flatten layer (matching current behavior)
    layer_list = [{"type": "flatten"}] + layer_list
    
    # Process each layer sequentially
    layers = []
    current_shape = input_shape
    
    for layer_dict in layer_list:
        layer, current_shape = layer_from_dict(layer_dict, current_shape)
        layers.append(layer)
    
    # Optionally append final linear layer if target_shape is provided
    if target_shape is not None:
        assert len(target_shape) == 1, "Target shape should be the output of a linear layer"
        if current_shape != target_shape:
            final_layer_dict = {
                "type": "linear",
                "out_features": target_shape[0],
                "bias": False
            }
            layer, current_shape = layer_from_dict(final_layer_dict, current_shape)
            layers.append(layer)
    
    return nn.Sequential(*layers), current_shape

class Network(nn.Module):
    def __init__(self, net, input_shape):
        super(Network, self).__init__()
        self.net = net
        self.input_shape = input_shape

    def init_hidden(self, batch_size=1):
        # Traverse the network
        device = next(self.net.parameters()).device
        v = torch.rand(batch_size,*self.input_shape).to(device)
        hidden_lst = []
        with torch.no_grad():
            for layer in self.net:
                if isinstance(layer, nn.GRUCell):
                    v = layer(v)
                    hidden_lst.append((torch.zeros_like(v),))
                elif isinstance(layer, nn.LSTMCell):
                    v, c = layer(v)
                    hidden_lst.append((torch.zeros_like(v), torch.zeros_like(c)))
                else:
                    v = layer(v)
                    hidden_lst.append(tuple())
        return hidden_lst

    def forward(self, v, hidden_lst=None):
        # Assumes v has shape (batch_size, *input_shape)
        hidden_lst = hidden_lst if hidden_lst is not None else self.init_hidden(batch_size=v.shape[0])
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.GRUCell):
                h = hidden_lst[i][0]
                v = layer(v, h)
                hidden_lst[i] = (v,)
            elif isinstance(layer, nn.LSTMCell):
                h, c = hidden_lst[i]
                v, c = layer(v, (h,c))
                hidden_lst[i] = (v, c)
            else:
                v = layer(v)
        return v, hidden_lst
    
    def get_output_shape(self):
        with torch.no_grad():
            v = torch.empty(1,*self.input_shape)
            v, hidden_lst = self.forward(v)
        return v.shape[1:]

    def __getitem__(self, idx):
        return self.net[idx]