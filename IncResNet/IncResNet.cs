using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace IncResNet;
/// <summary>
/// Custom version of IncResNet for 200x200 images.
/// </summary>
public class IncResNet : Module<Tensor, Tensor> {
    private readonly Module<Tensor, Tensor> _layers;

    // Predifined settings for deep regresive model.
    public static IncResNet IncResNetv1(Device? device = null, string? weights = null) {
        var model = new IncResNet("IncResNetv1", new int[] { 4, 8, 4 }, 1, device);
        if (weights != null) {
            model.load(weights);
        }
        return model;
    }

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

    private static void MakeLayer(string name, List<(string, Module<Tensor, Tensor>)> modules, int filters, int numModules) {
        for (int i = 0; i < numModules; i++) {
            modules.Add(($"{name}-incres-{filters}-{i}", new InceptionResidualBlock($"{name}-incres-{filters}-{i}", filters)));
        }
    }

    public override Tensor forward(Tensor input) {
        return _layers.forward(input);
    }

    private class StemBlock : Module<Tensor, Tensor> {
        private readonly Module<Tensor, Tensor> _layers;

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

        public override Tensor forward(Tensor input) {
            return _layers.forward(input);
        }
    }

    private class ReductionBlock : Module<Tensor, Tensor> {
        private readonly Module<Tensor, Tensor> _convolution;
        private readonly Module<Tensor, Tensor> _pooling;

        public ReductionBlock(string name, long filters) : base(name) {
            _convolution = Sequential(new List<(string, Module<Tensor, Tensor>)> {
                ($"{name}-conv2d-1", Conv2d(inputChannel: filters, outputChannel: filters, kernelSize: 3, stride: 2)),
                ($"{name}-bnrm2d-1", BatchNorm2d(features: filters)),
                ($"{name}-relu-1", ReLU(inplace: true))
            });
            _pooling = MaxPool2d(kernelSize: 3, stride: 2);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input) {
            return cat(tensors: new List<Tensor> { _convolution.forward(input), _pooling.forward(input) }, dim: 1);
        }
    }

    private class InceptionResidualBlock : Module<Tensor, Tensor> {
        private readonly Module<Tensor, Tensor> _module1x1;
        private readonly Module<Tensor, Tensor> _module3x3;
        private readonly Module<Tensor, Tensor> _module5x5;

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

        public override Tensor forward(Tensor input) {
            Tensor x = cat(tensors: new List<Tensor> {
                _module1x1.forward(input), _module3x3.forward(input), _module5x5.forward(input)
            }, dim: 1);
            return x.add_(input).relu_();
        }
    }
}