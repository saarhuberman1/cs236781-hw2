import torch
import torch.nn as nn
import itertools as it
from typing import Sequence

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """
    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        # raise NotImplementedError()
        channels_list = [in_channels] + list(self.channels)
        N = len(self.channels)
        P = self.pool_every

        for i in range(N):
            layers.append(nn.Conv2d(channels_list[i],
                                    channels_list[i+1],
                                    *self.conv_params.values()))
            layers.append(ACTIVATIONS[self.activation_type](*self.activation_params.values()))
            if (i+1) % P == 0:
                layers.append(POOLINGS[self.pooling_type](*self.pooling_params.values()))


        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """

        def conv_size(h_in, w_in):
            if "kernel_size" in self.conv_params.keys():
                kernel_size = (self.conv_params['kernel_size'], self.conv_params['kernel_size']) if isinstance(
                    self.conv_params['kernel_size'], int) else self.conv_params['kernel_size']
            else:
                kernel_size = (self.kernel_size, self.kernel_size) if isinstance(
                    self.kernel_size, int) else self.kernel_size

            if "padding" in self.conv_params.keys():
                padding = (self.conv_params['padding'], self.conv_params['padding']) if isinstance(
                    self.conv_params['padding'], int) else self.conv_params['padding']
            else:
                padding = (0,0)

            if "dilation" in self.conv_params.keys():
                dilation = (self.conv_params['dilation'], self.conv_params['dilation']) if isinstance(
                    self.conv_params['dilation'], int) else self.conv_params['dilation']
            else:
                dilation = (1,1)

            if "stride" in self.conv_params.keys():
                stride = (self.conv_params['stride'], self.conv_params['stride']) if isinstance(
                    self.conv_params['stride'], int) else self.conv_params['stride']
            else:
                stride = (1,1)

            # print("========convo===========")
            # print("kernel = " + str(kernel_size))
            # print("padding = " + str(padding))
            # print("dilation = " + str(dilation))
            # print("stride = " + str(stride))

            h_out = floor(((h_in + 2 *padding[0] - dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1)
            w_out = floor(((w_in + 2 *padding[1] - dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1)
            # print(h_out , w_out)
            return h_out,w_out

        def pool_size(h_in, w_in):

            if "kernel_size" in self.pooling_params.keys():
                kernel_size = (self.pooling_params['kernel_size'], self.pooling_params['kernel_size']) if isinstance(
                    self.pooling_params['kernel_size'], int) else self.pooling_params['kernel_size']
            # else:
            #     kernel_size = (self.kernel_size, self.kernel_size) if isinstance(
            #         self.kernel_size, int) else self.kernel_size

            if "padding" in self.pooling_params.keys():
                padding = (self.pooling_params['padding'], self.pooling_params['padding']) if isinstance(
                    self.pooling_params['padding'], int) else self.pooling_params['padding']
            else:
                padding = (0,0)

            if "dilation" in self.pooling_params.keys():
                dilation = (self.pooling_params['dilation'], self.pooling_params['dilation']) if isinstance(
                    self.pooling_params['dilation'], int) else self.pooling_params['dilation']
            else:
                dilation = (1,1)

            if "stride" in self.pooling_params.keys():
                stride = (self.pooling_params['stride'], self.pooling_params['stride']) if isinstance(
                    self.pooling_params['stride'], int) else self.pooling_params['stride']
            else:
                stride = kernel_size

            # print("========pooling===========")
            # print("kernel = " + str(kernel_size))
            # print("padding = " + str(padding))
            # print("dilation = " + str(dilation))
            # print("stride = " + str(stride))

            h_out = floor(((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
            w_out = floor(((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
            # print(h_out , w_out)
            return h_out, w_out

        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            _, in_h, in_w, = tuple(self.in_size)

            from math import floor

            for channel in range(len(self.channels)):
                # print(channel)
                in_h, in_w = conv_size(in_h, in_w)
                if (channel+1) % self.pool_every == 0:
                # if channel > 0 and channel % self.pool_every == 0:
                    in_h, in_w = pool_size(in_h, in_w)

            return in_h * in_w * self.channels[-1]




            # return self.channels[-1] * ceil((in_h * in_w) / ((self.pooling_params['kernel_size'] * self.pooling_params['kernel_size']) ** (len(self.channels) // self.pool_every)))
            # return self.channels[-1] * self.conv_params['kernel_size'] * self.conv_params['kernel_size']

            # raise NotImplementedError()
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_classifier(self):
        layers = []

        # Discover the number of features after the CNN part.
        n_features = self._n_features()
        # print("n features = " + str(n_features))

        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        fc_dims = [n_features] + list(self.hidden_dims)
        for i in range(len(fc_dims)-1):
            layers.append(nn.Linear(fc_dims[i], fc_dims[i+1]))
            layers.append(ACTIVATIONS[self.activation_type](*self.activation_params.values()))
        # ========================
        layers.append(nn.Linear(fc_dims[-1], self.out_classes))

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        # print(features.shape)
        features = features.view(features.size(0), -1)
        # print(features.shape)
        # note: no need to reshape the features now
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.in_channels = in_channels
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activation_type = activation_type
        self.activation_params = activation_params

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        self.main_path = self._create_main_path()
        self.shortcut_path = self._create_shortcut_path()
        # ========================

    def _create_main_path(self):
        layers = []
        channels_list = [self.in_channels] + list(self.channels)
        N = len(self.channels)

        for i in range(N-1):
            layer_padding = int((self.kernel_sizes[i] - 1) / 2)  # the kernel size is always odd (given)
            layers.append(nn.Conv2d(channels_list[i],
                                    channels_list[i+1],
                                    kernel_size=self.kernel_sizes[i],
                                    bias=True,
                                    padding=layer_padding))
            if self.dropout:
                layers.append(nn.Dropout2d(self.dropout))
            if self.batchnorm:
                layers.append(nn.BatchNorm2d(channels_list[i+1]))
            layers.append(ACTIVATIONS[self.activation_type](*self.activation_params.values()))
        layer_padding = int((self.kernel_sizes[-1] - 1) / 2)
        layers.append(nn.Conv2d(channels_list[-2],
                                channels_list[-1],
                                kernel_size=self.kernel_sizes[-1],
                                bias=True,
                                padding=layer_padding))

        seq = nn.Sequential(*layers)
        return seq

    def _create_shortcut_path(self):
        if self.in_channels == self.channels[-1]:
            layers = [nn.Identity()]
        else:
            layers = [nn.Conv2d(self.in_channels, self.channels[-1], bias=False, kernel_size=1, padding=0)]
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        # ====== YOUR CODE: ======
        in_channels = in_out_channels
        channels = [inner_channels[0]] + list(inner_channels) + [in_out_channels]
        kernels= [1] + list(inner_kernel_sizes) + [1]
        ResidualBlock.__init__(self,
                               in_channels=in_channels,
                               channels=channels,
                               kernel_sizes=kernels,
                               **kwargs)

        # ========================


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )


    def _make_feature_extractor(self):
        self.conv_params['kernel_size'] = 3
        self.conv_params['padding'] = int((3 - 1) / 2)

        # print(self.pooling_params)

        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======

        channels_list = [in_channels] + list(self.channels)
        N = len(self.channels)
        P = self.pool_every

        for i in range(0, N, P):
            channels = channels_list[i+1:i+P+1]

            res_block = ResidualBlock(channels_list[i],
                                      channels,
                                      kernel_sizes=[3]*len(channels),
                                      batchnorm=self.batchnorm,
                                      dropout=self.dropout,
                                      activation_type=self.activation_type,
                                      activation_params=self.activation_params
                                      )
            layers.append(res_block)
            layers.append(ACTIVATIONS[self.activation_type](*self.activation_params.values()))

            # TODO - MARWA: should it be <= ??? - controlling whether we have a pooling layer as the last layer
            if i + P <= N:
                layers.append(POOLINGS[self.pooling_type](*self.pooling_params.values()))

        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, *args, **kwargs):
        """
        See ConvClassifier.__init__
        """
        super().__init__(*args, **kwargs)

        # TODO: Add any additional initialization as needed.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    pass
    # ========================
