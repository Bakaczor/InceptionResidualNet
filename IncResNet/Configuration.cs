using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
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
    private static int _epochs = 25;

    private static readonly string weightsPath = @".\model\weights.pt";
    private static readonly string trainPath = @"C:\Users\Bakaczor\PycharmProjects\dataAugmentation\final\training";
    private static readonly string testPath = @"C:\Users\Bakaczor\PycharmProjects\dataAugmentation\final\test";

    private readonly static int _loggingInterval = 10; // for logging frequency

    private readonly static int _timeout = 3600; // by default an hour

    public static void Start(string[] args) {

        random.manual_seed(1);

        var device = cuda.is_available() ? CUDA : CPU;

        if (device.type == DeviceType.CUDA) {
            _trainBatchSize *= 4;
            _testBatchSize *= 4;
            _epochs *= 4;
        }

        Console.WriteLine($"\tCreating the model...");

        var weights = args.Length > 1 ? args[1] : null;
        var epochs = args.Length > 2 ? int.Parse(args[2]) : _epochs;
        var timeout = args.Length > 3 ? int.Parse(args[3]) : _timeout;

        Module<Tensor, Tensor> model = IncResNet.IncResNetv1(device, weights);

        Console.WriteLine($"\tRunning IncResNetv1 on {device.type} for {epochs} epochs, " +
            $"terminating after {TimeSpan.FromSeconds(timeout)}.");

        Console.WriteLine($"\tPreparing training and test data...");
        Console.WriteLine();

        using var trainData = new ImageDataset(trainPath);
        using var testData = new ImageDataset(testPath);

        using var train = DataLoader(trainData, _trainBatchSize, true, device);
        using var test = DataLoader(testData, _testBatchSize, false, device);

        using var optimizer = optim.Adam(model.parameters());

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

        if (weights != null) {
            model.save(weights);
        } else {
            model.save(weightsPath);
        }
        model.Dispose();
    }

    private static void Train(
        Module<Tensor, Tensor> model,
        optim.Optimizer optimizer,
        Loss<Tensor, Tensor, Tensor> loss,
        DataLoader dataLoader,
        int epoch,
        long size) {

        model.train();

        int batchId = 1;
        long total = 0;
        long correct = 0;

        Console.WriteLine($"Epoch: {epoch}...");

        using var d = NewDisposeScope();
        foreach (var data in dataLoader) {

            optimizer.zero_grad();

            var target = data["label"];
            var prediction = model.call(data["data"]);
            var lsm = softmax(prediction, 1);
            var output = loss.call(lsm.mean(new long[] { 1 }), target);

            output.backward();

            optimizer.step();

            total += target.shape[0];

            var predicted = prediction.mean(new long[] { 1 });
            correct += predicted.to_type(int32).equal(target.to_type(int32)).sum().ToInt32();

            if (batchId % _loggingInterval == 0 || total == size) {
                Console.WriteLine($"\rTrain: epoch {epoch} [{total} / {size}] Loss: {output.ToSingle():0.000000} | " +
                    $"Accuracy: {((float)correct / total):0.000000}");
            }
            batchId++;

            d.DisposeEverything();
        }
    }

    private static void Test(
        Module<Tensor, Tensor> model,
        Loss<Tensor, Tensor, Tensor> loss,
        DataLoader dataLoader,
        long size) {

        model.eval();

        double testLoss = 0;
        long correct = 0;
        int batchCount = 0;

        using var d = NewDisposeScope();
        foreach (var data in dataLoader) {

            var target = data["label"];
            var prediction = model.call(data["data"]);
            var lsm = softmax(prediction, 1);
            var output = loss.call(lsm.mean(new long[] { 1 }), target);

            testLoss += output.ToSingle();
            batchCount += 1;

            var predicted = prediction.mean(new long[] { 1 });
            correct += predicted.to_type(int32).equal(target.to_type(int32)).sum().ToInt32();

            d.DisposeEverything();
        }
        Console.WriteLine($"\rTest set: Average loss {(testLoss / batchCount):0.0000} | " +
            $"Accuracy {((float)correct / size):0.0000}");
    }
}