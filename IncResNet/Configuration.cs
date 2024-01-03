using System.Diagnostics;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;

namespace IncResNet;
/// <summary>
/// Custom configuration for training and evaluating models operating on 200x200 image datasets
/// </summary>
public static class Configuration {
    private static int _trainBatchSize = 32;
    private static int _testBatchSize = 64;
    private static int _epochs = 25;

    private static readonly string weightsPath = @".\weights.bin";
    private static readonly string trainPath = @"C:\Users\Bakaczor\PycharmProjects\dataAugmentation\final\training";
    private static readonly string testPath = @"C:\Users\Bakaczor\PycharmProjects\dataAugmentation\final\test";

    private readonly static int _loggingInterval = 20; // for logging frequency

    private readonly static int _timeout = 3 * 3600; // max training time

    public static void Start(string[] args) {

        random.manual_seed(1);

        var device = cuda.is_available() ? CUDA : CPU;

        if (device.type == DeviceType.CUDA) {
            _trainBatchSize *= 4;
            _testBatchSize *= 4;
            _epochs *= 4;
        }

        Console.WriteLine($"\tCreating the model...");

        var weights = args.Length > 0 ? args[0] : null;
        if (weights != null) {
            Console.WriteLine("\tWeights loaded...");
        } else {
            Console.WriteLine("\tWeights not loaded...");
        }
        var epochs = args.Length > 1 ? int.Parse(args[1]) : _epochs;
        var timeout = args.Length > 2 ? int.Parse(args[2]) : _timeout;

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
            var output = loss.call(prediction, target);
            output.backward();

            optimizer.step();

            total += target.shape[0];
            correct += prediction.to_type(int32).eq_(target.to_type(int32)).sum().ToInt32();

            if (batchId % _loggingInterval == 0 || total == size) {
                double lossSq = Math.Sqrt(output.ToSingle());
                Console.WriteLine($"\r[Train] Epoch: {epoch} [{total} / {size}] MSE Root: {lossSq:0.000000} | " +
                    $"Total perfect prediction: {((float)correct / total):0.000000}");
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
            var output = loss.call(prediction, target);

            testLoss += output.ToSingle();
            batchCount += 1;

            correct += prediction.to_type(int32).eq_(target.to_type(int32)).sum().ToInt32();

            d.DisposeEverything();
        }
        Console.WriteLine($"\r[Test] Average MSE Root: {(Math.Sqrt(testLoss) / batchCount):0.0000} | " +
            $"Total perfect prediction: {((float)correct / size):0.0000}");
    }
}