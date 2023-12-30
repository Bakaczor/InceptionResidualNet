using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.torch.utils.data;

namespace IncResNet; 
/// <summary>
/// Custom configuration for training and evaluating models operating on 200x200 image datasets
/// </summary>
public static class Configuration {
    private static int _trainBatchSize = 32;
    private static int _testBatchSize = 64;
    private static int _epochs = 128;

    private readonly static int _loggingInterval = 25; // for logging frequency
    private readonly static int _numClasses = 111; // number of different ages in dataset

    private readonly static int _timeout = 3600; // by default an hour

    public static void Main(string[] args) {
        string datasetPath = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);

        torch.random.manual_seed(1);

        var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;

        if (device.type == DeviceType.CUDA) {
            _trainBatchSize *= 8;
            _testBatchSize *= 8;
            _epochs *= 8;
        }

        Console.WriteLine($"\tCreating the model...");
        Module<torch.Tensor, torch.Tensor> model = IncResNet.IncResNetv1(device);

        // override epochs and timeout with console input
        var epochs = args.Length > 1 ? int.Parse(args[1]) : _epochs;
        var timeout = args.Length > 2 ? int.Parse(args[2]) : _timeout;

        Console.WriteLine($"\tRunning IncResNetv1 on {device.type} for {epochs} epochs, " +
            $"terminating after {TimeSpan.FromSeconds(timeout)}.");

        Console.WriteLine($"\tPreparing training and test data...");
        Console.WriteLine();

        // TODO: import nuget and load images
        using (Dataset trainData = torchvision.datasets.ImageFolder(datasetPath, true, download: true),
                       testData = torchvision.datasets.CIFAR100(datasetPath, false, download: true)) {
            using var train = DataLoader(trainData, _trainBatchSize, device: device, shuffle: true);
            using var test = DataLoader(testData, _testBatchSize, device: device, shuffle: false);

            using var optimizer = torch.optim.Adam(model.parameters());

            var totalSW = new Stopwatch();
            totalSW.Start();

            for (int epoch = 1; epoch <= epochs; epoch++) {

                var epochSW = new Stopwatch();
                epochSW.Start();

                Train(model, optimizer, MSELoss(), train, epoch, trainData.Count);
                Test(model, MSELoss(), test, testData.Count);

                epochSW.Stop();
                Console.WriteLine($"Elapsed time for this epoch: {epochSW.Elapsed.TotalSeconds}s.");

                if (totalSW.Elapsed.TotalSeconds > timeout) { break; }
            }

            totalSW.Stop();
            Console.WriteLine($"Elapsed training time: {totalSW.Elapsed}s.");
        }
        model.Dispose();
    }

    private static void Train(
        Module<torch.Tensor, torch.Tensor> model,
        torch.optim.Optimizer optimizer,
        Loss<torch.Tensor, torch.Tensor, torch.Tensor> loss,
        DataLoader dataLoader,
        int epoch,
        long size) {

        model.train();

        int batchId = 1;
        long total = 0;
        long correct = 0;

        Console.WriteLine($"Epoch: {epoch}...");

        using var d = torch.NewDisposeScope();
        foreach (var data in dataLoader) {

            optimizer.zero_grad();

            var target = data["label"];
            var prediction = model.call(data["data"]);
            var lsm = softmax(prediction, 1);
            var output = loss.call(lsm, target);

            output.backward();

            optimizer.step();

            total += target.shape[0];

            var predicted = prediction.argmax(1);
            correct += predicted.eq(target).sum().ToInt64();

            if (batchId % _loggingInterval == 0 || total == size) {
                Console.WriteLine($"\rTrain: epoch {epoch} [{total} / {size}] Loss: {output.ToSingle():0.000000} | " +
                    $"Accuracy: {((float)correct / total):0.000000}");
            }
            batchId++;

            d.DisposeEverything();
        }
    }

    private static void Test(
        Module<torch.Tensor, torch.Tensor> model,
        Loss<torch.Tensor, torch.Tensor, torch.Tensor> loss,
        DataLoader dataLoader,
        long size) {

        model.eval();

        double testLoss = 0;
        long correct = 0;
        int batchCount = 0;

        using var d = torch.NewDisposeScope();
        foreach (var data in dataLoader) {

            var target = data["label"];
            var prediction = model.call(data["data"]);
            var lsm = softmax(prediction, 1);
            var output = loss.call(lsm, target);

            testLoss += output.ToSingle();
            batchCount += 1;

            var predicted = prediction.argmax(1);
            correct += predicted.eq(target).sum().ToInt64();

            d.DisposeEverything();
        }
        Console.WriteLine($"\rTest set: Average loss {(testLoss / batchCount):0.0000} | " +
            $"Accuracy {((float)correct / size):0.0000}");
    }
}