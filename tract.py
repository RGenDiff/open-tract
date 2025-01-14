import torch
import torch.nn as nn
from torch.linalg import inv

def unfold3d(x, kernel_size, padding=0, stride=1, dilation=1):
    """
    Perform a 3D unfold operation on a 5D tensor.

    Args:
        x: A 5D tensor of shape (batch_size, channels, depth, height, width).
        kernel_size: A tuple of 3 integers representing the kernel size in each dimension.
        padding: A tuple of 3 integers representing the padding in each dimension.
        stride: A tuple of 3 integers representing the stride in each dimension.
        dilation: A tuple of 3 integers representing the dilation in each dimension.
    """

    # Extract dimensions
    batch_size, channels, depth, height, width = x.size()

    # Apply padding
    if padding:
        x = nn.functional.pad(x, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]))

    # Unfold in the depth dimension
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.unfold(4, kernel_size[2], stride[2])

    # Permute dimensions to arrange the kernel elements in the channel dimension
    # New shape: (B, C, out_d, out_h, out_w, kD, kH, kW)
    x = x.permute(0, 1, 5, 6, 7, 2, 3, 4).contiguous()

    # Reshape to combine kernel elements into the channel dimension
    # New shape: (B, C * kD * kH * kW, out_d * out_h * out_w)
    x = x.view(batch_size, channels * kernel_size[0] * kernel_size[1] * kernel_size[2], -1)

    return x

class TrActFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, lambda_, is_conv, conv_params):
        """
        Custom forward pass for TrACT.

        Args:
            input (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight parameter.
            bias (torch.Tensor): Bias parameter.
            lambda_ (float): Regularization parameter.
            is_conv (bool): Whether the layer is a convolutional layer.
            conv_params (dict): Convolutional parameters (stride, padding, dilation, groups).

        Returns:
            torch.Tensor: The output of the layer.
        """
        ctx.save_for_backward(input, weight, bias)
        ctx.lambda_ = lambda_
        ctx.is_conv = is_conv
        ctx.conv_params = conv_params

        if is_conv:
            stride, padding, dilation, groups, dim = conv_params
            if dim == 1:
                output = torch.nn.functional.conv1d(input, weight, bias, stride, padding, dilation, groups)
            elif dim == 2:
                output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
            elif dim == 3:
                output = torch.nn.functional.conv3d(input, weight, bias, stride, padding, dilation, groups)
            else:
                raise ValueError(f"Unsupported convolution dimension: {dim}")
        else:
            output = input @ weight.T
            if bias is not None:
                output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        is_conv = ctx.is_conv
        conv_params = ctx.conv_params

        # cast the inputs to float
        grad_output = grad_output.float()
        input = input.float()

        if is_conv:
            # Unpack convolutional parameters
            stride, padding, dilation, groups, dim = conv_params
            kernel_size = weight.shape[2:]  # Kernel shape (kW) or (kH, kW) or (kD, kH, kW)

            if dim == 1:
                input_unfolded = torch.nn.functional.unfold(input.unsqueeze(-1), kernel_size=(kernel_size[0], 1),
                                                            dilation=(dilation[0], 1), padding=(padding[0], 0),
                                                            stride=(stride[0], 1)).squeeze(-1)
            elif dim == 2:
                input_unfolded = torch.nn.functional.unfold(input, kernel_size, dilation, padding, stride)
            elif dim == 3:
                input_unfolded = unfold3d(input, kernel_size=kernel_size, dilation=dilation,
                                                            padding=padding, stride=stride)
            else:
                raise ValueError(f"Unsupported convolution dimension: {dim}")

            # Flatten grad_output for weight gradient computation
            grad_output_unfolded = grad_output.permute(0, *range(2, 2 + dim), 1).reshape(-1, grad_output.shape[1])

            # Prepare input_unfolded for TrACT adjustment
            input_unfolded_flat = input_unfolded.permute(0, 2, 1).reshape(-1, input_unfolded.shape[1])

            # TrAct adjustment
            b, n = input_unfolded_flat.shape
            reg_term = ctx.lambda_ * torch.eye(n, device=input.device)
            xTx = input_unfolded_flat.T @ input_unfolded_flat / b
            # note inv only supports FP32
            inv_arg = xTx + reg_term
            inv_term = torch.linalg.inv(inv_arg.float())

            # Compute TrAct-adjusted weight gradient
            grad_weight = grad_output_unfolded.T @ input_unfolded_flat @ inv_term
            grad_weight = grad_weight.view(weight.shape)  # Reshape back to original weight shape
            
            # Compute bias gradient
            grad_bias = grad_output.sum(dim=(0, *range(2, 2 + dim))) if bias is not None else None

        else:
            # Handle B, *, C for Linear
            input_flat = input.view(-1, input.shape[-1])  # Flatten to (B*, C)
            grad_output_flat = grad_output.view(-1, grad_output.shape[-1])  # Flatten to (B*, M)

            b, n = input_flat.shape  # Batch size and input features
            reg_term = ctx.lambda_ * torch.eye(n, device=input.device)
            xTx = input_flat.T @ input_flat / b
            # note inv only supports FP32
            inv_arg = xTx + reg_term
            inv_term = torch.linalg.inv(inv_arg.float())

            grad_weight = grad_output_flat.T @ input_flat @ inv_term
                
            grad_bias = grad_output_flat.sum(0) if bias is not None else None

        # First layer, no need to propagate grad_input
        grad_input = None
        return grad_input, grad_weight, grad_bias, None, None, None


class TrAct(nn.Module):

    def __init__(self, module, lambda_=0.1):
        """
        Wraps a given nn.Linear or nn.Conv* module and modifies its backward pass using TrACT.

        Args:
            module (nn.Module): The module to wrap (must be nn.Linear or nn.Conv*).
            lambda_ (float): The regularization parameter for TrACT.
        """
        super().__init__()
        if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            raise TypeError("TrAct only supports nn.Linear or convolutional layers.")

        self.lambda_ = lambda_

        # Transfer weight and bias to the TrACT wrapper directly
        self.weight = module.weight
        self.bias = module.bias if hasattr(module, "bias") else None

        # Handle convolution parameters for Conv layers
        self.is_conv = isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        if self.is_conv:
            self.stride = module.stride
            self.padding = module.padding
            self.dilation = module.dilation
            self.groups = module.groups
            self.dim = len(module.weight.shape) - 2  # Determine dimension (1D, 2D, or 3D)

    def forward(self, x):
        if self.is_conv:
            conv_params = (self.stride, self.padding, self.dilation, self.groups, self.dim)
            output = TrActFunction.apply(x, self.weight, self.bias, self.lambda_, True, conv_params)
        else:
            output = TrActFunction.apply(x, self.weight, self.bias, self.lambda_, False, None)
        return output