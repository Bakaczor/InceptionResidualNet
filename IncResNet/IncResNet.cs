using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace IncResNet;
/// <summary>
/// Custom version of IncResNet for 200x200 images.
/// </summary>
public class IncResNet : Module<Tensor, Tensor> {
    private readonly Module<Tensor, Tensor> _layers;
    /// <summary>
    /// Creates a predefined version of IncResNetv1 with specified device and weights.
    /// </summary>
    /// <param name="device">The device to run the model on (e.g., CPU or CUDA).</param>
    /// <param name="weights">The path to the weights file to load.</param>
    /// <returns>An instance of the IncResNet model.</returns>
    public static IncResNet IncResNetv1(Device? device = null, string? weights = null) {
        var model = new IncResNet("IncResNetv1", new int[] { 4, 8, 4 }, 1, device);
        if (weights != null) {
            model.load(weights);
        }
        return model;
    }
    /// <summary>
    /// Initializes a new instance of the <see cref="IncResNet"/> class.
    /// </summary>
    /// <param name="name">The name of the model.</param>
    /// <param name="numModules">The number of modules in each layer.</param>
    /// <param name="resultSize">The size of the output.</param>
    /// <param name="device">The device to run the model on (e.g., CPU or CUDA).</param>
    public IncResNet(string name, IList<int> numModules, int resultSize, Device? device = null) : base(name) {
        var modules = new List<(string, Module<Tensor, Tensor>)> {
            ($"{name}-stem", new StemBlock($"{name}-stem")),
            ($"{name}-red-1", new ReductionBlock($"{name}-red-1", 64))
        };

        MakeLayer(name, modules, 128, numModules[0]);
        modules.Add(($"{name}-red-2", new ReductionBlock($"{name}-red-2", 128)));
        MakeLayer(name, modules, 256, numModules[1]);
        modules.Add(($"{name}-red-3", new ReductionBlock($"{name}-red-3", 256)));
        MakeLayer(name, modules, 512, numModules[2]);
        modules.Add(($"{name}-red-4", new ReductionBlock($"{name}-red-4", 512)));

        modules.Add(($"{name}-avgpool", AvgPool2d(new long[] { 5, 5 })));
        modules.Add(($"{name}-flatten", Flatten()));
        modules.Add(($"{name}-dropout", Dropout(p: 0.25, inplace: true)));
        modules.Add(($"linear", Linear(inputSize: 1024, resultSize)));

        _layers = Sequential(modules);

        RegisterComponents();

        if (device != null && device.type == DeviceType.CUDA) {
            this.to(device);
        }
    }
    /// <summary>
    /// Adds layers to the model.
    /// </summary>
    /// <param name="name">The name of the model.</param>
    /// <param name="modules">The list of modules to add layers to.</param>
    /// <param name="filters">The number of filters for the layers.</param>
    /// <param name="numModules">The number of modules to add.</param>
    private static void MakeLayer(string name, List<(string, Module<Tensor, Tensor>)> modules, int filters, int numModules) {
        for (int i = 0; i < numModules; i++) {
            modules.Add(($"{name}-incres-{filters}-{i}", new InceptionResidualBlock($"{name}-incres-{filters}-{i}", filters)));
        }
    }
    /// <summary>
    /// Performs a forward pass of the input tensor through the model.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor.</returns>
    public override Tensor forward(Tensor input) {
        return _layers.forward(input);
    }
    /// <summary>
    /// A block that forms the initial layers of the network.
    /// </summary>
    private class StemBlock : Module<Tensor, Tensor> {
        private readonly Module<Tensor, Tensor> _layers;
        /// <summary>
        /// Initializes a new instance of the <see cref="StemBlock"/> class.
        /// </summary>
        /// <param name="name">The name of the block.</param>
        public StemBlock(string name) : base(name) {
            _layers = Sequential(new List<(string, Module<Tensor, Tensor>)> {
                ($"{name}-conv2d-1", Conv2d(inputChannel: 3, outputChannel: 32, kernelSize: 3, stride: 2)),
                ($"{name}-bnrm2d-1", BatchNorm2d(32)),
                ($"{name}-relu-1", ReLU(inplace: true)),
                ($"{name}-conv2d-2", Conv2d(inputChannel: 32, outputChannel: 32, kernelSize: 3)),
                ($"{name}-bnrm2d-2", BatchNorm2d(32)),
                ($"{name}-relu-2", ReLU(inplace: true)),
                ($"{name}-conv2d-3", Conv2d(inputChannel: 32, outputChannel: 64, kernelSize: 3, padding: 1)),
                ($"{name}-bnrm2d-3", BatchNorm2d(64)),
                ($"{name}-relu-3", ReLU(inplace: true))
            });

            RegisterComponents();
        }
        /// <summary>
        /// Performs a forward pass of the input tensor through the stem block.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public override Tensor forward(Tensor input) {
            return _layers.forward(input);
        }
    }
    /// <summary>
    /// A block that performs reduction operations within the network.
    /// </summary>
    private class ReductionBlock : Module<Tensor, Tensor> {
        private readonly Module<Tensor, Tensor> _convolution;
        private readonly Module<Tensor, Tensor> _pooling;
        /// <summary>
        /// Initializes a new instance of the <see cref="ReductionBlock"/> class.
        /// </summary>
        /// <param name="name">The name of the block.</param>
        /// <param name="filters">The number of filters for the convolution layers.</param>
        public ReductionBlock(string name, long filters) : base(name) {
            _convolution = Sequential(new List<(string, Module<Tensor, Tensor>)> {
                ($"{name}-conv2d-1", Conv2d(inputChannel: filters, outputChannel: filters, kernelSize: 3, stride: 2)),
                ($"{name}-bnrm2d-1", BatchNorm2d(features: filters)),
                ($"{name}-relu-1", ReLU(inplace: true))
            });
            _pooling = MaxPool2d(kernelSize: 3, stride: 2);
            RegisterComponents();
        }
        /// <summary>
        /// Performs a forward pass of the input tensor through the reduction block.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public override Tensor forward(Tensor input) {
            return cat(tensors: new List<Tensor> { _convolution.forward(input), _pooling.forward(input) }, dim: 1);
        }
    }
    /// <summary>
    /// A block that performs inception and residual operations within the network.
    /// </summary>
    private class InceptionResidualBlock : Module<Tensor, Tensor> {
        private readonly Module<Tensor, Tensor> _module1x1;
        private readonly Module<Tensor, Tensor> _module3x3;
        private readonly Module<Tensor, Tensor> _module5x5;
        /// <summary>
        /// Initializes a new instance of the <see cref="InceptionResidualBlock"/> class.
        /// </summary>
        /// <param name="name">The name of the block.</param>
        /// <param name="filters">The number of filters for the convolution layers.</param>
        public InceptionResidualBlock(string name, long filters) : base(name) {
            long filters4 = filters / 4;
            long filters2 = filters / 2;

            _module1x1 = Sequential(new List<(string, Module<Tensor, Tensor>)> {
                ($"{name}-conv2d-1", Conv2d(inputChannel: filters, outputChannel: filters4, kernelSize: 1)),
                ($"{name}-bnrm2d-1", BatchNorm2d(features: filters4)),
                ($"{name}-relu-1", ReLU(inplace: true))
            });

            _module3x3 = Sequential(new List<(string, Module<Tensor, Tensor>)> {
                ($"{name}-conv2d-1", Conv2d(inputChannel: filters, outputChannel: filters4, kernelSize: 1)),
                ($"{name}-bnrm2d-1", BatchNorm2d(features: filters4)),
                ($"{name}-relu-1", ReLU(inplace: true)),
                ($"{name}-conv2d-2", Conv2d(inputChannel : filters4, outputChannel: filters4, kernelSize: 3, padding: 1)),
                ($"{name}-bnrm2d-2", BatchNorm2d(features: filters4)),
                ($"{name}-relu-2", ReLU(inplace: true))
            });

            _module5x5 = Sequential(new List<(string, Module<Tensor, Tensor>)> {
                ($"{name}-conv2d-1", Conv2d(inputChannel: filters, outputChannel: filters4, kernelSize: 1)),
                ($"{name}-bnrm2d-1", BatchNorm2d(features: filters4)),
                ($"{name}-relu-1", ReLU(inplace: true)),
                ($"{name}-conv2d-2", Conv2d(inputChannel : filters4, outputChannel: filters4, kernelSize: 3, padding: 1)),
                ($"{name}-bnrm2d-2", BatchNorm2d(features: filters4)),
                ($"{name}-relu-2", ReLU(inplace: true)),
                ($"{name}-conv2d-3", Conv2d(inputChannel : filters4, outputChannel: filters2, kernelSize: 3, padding: 1)),
                ($"{name}-bnrm2d-3", BatchNorm2d(features: filters2)),
                ($"{name}-relu-3", ReLU(inplace: true))
            });

            RegisterComponents();
        }
        /// <summary>
        /// Performs a forward pass of the input tensor through the inception residual block.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public override Tensor forward(Tensor input) {
            Tensor x = cat(tensors: new List<Tensor> {
                _module1x1.forward(input), _module3x3.forward(input), _module5x5.forward(input)
            }, dim: 1);
            return x.add_(input).relu_();
        }
    }
}