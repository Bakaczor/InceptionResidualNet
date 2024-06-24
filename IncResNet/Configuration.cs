using System.Diagnostics;
using System.Text;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;

namespace IncResNet;
/// <summary>
/// Custom configuration for training and testing models operating on 200x200 image datasets.
/// </summary>
public static class Configuration {
    private static int _trainBatchSize = 32;
    private static int _otherBatchSize = 64;
    private static int _epochs = 5;

    private static readonly string _weightsSavePath = @".\weights.bin";
    private static readonly string _trainLoadPath = @"D:\workspace\dataset\training";
    private static readonly string _validationLoadPath = @"D:\workspace\dataset\validation";
    private static readonly string _testLoadPath = @"D:\workspace\dataset\test";

    private static readonly int _loggingInterval = 20; // logging frequency
    private static readonly int _timeout = 2 * 3600; // max training time

    private static readonly string _trainOutput = @".\train.txt";
    private static readonly string _validationOutput = @".\validation.txt";
    private static readonly string _testOutput = @".\test.txt";

    private static readonly StringBuilder _trainStrings = new();
    private static readonly StringBuilder _validationStrings = new();
    private static readonly StringBuilder _testStrings = new();
    /// <summary>
    /// Starts the training or testing process based on the specified parameters.
    /// </summary>
    /// <param name="args">An array of command-line arguments: weights file path, number of epochs, timeout duration.</param>
    /// <param name="training">A boolean indicating whether to run in training mode or testing mode.</param>
    public static void Start(string[] args, bool training) {

        random.manual_seed(1);

        var device = cuda.is_available() ? CUDA : CPU;

        if (device.type == DeviceType.CUDA) {
            _trainBatchSize *= 4;
            _otherBatchSize *= 4;
            _epochs *= 4;
        }

        Console.WriteLine($"\tCreating the model...");

        string? weights = args.Length > 0 ? args[0] : null;
        if (weights != null) {
            Console.WriteLine("\tWeights loaded...");
        } else {
            Console.WriteLine("\tWeights not loaded...");
        }
        // overwrite epochs and timeout from console
        int epochs = args.Length > 1 ? int.Parse(args[1]) : _epochs;
        int timeout = args.Length > 2 ? int.Parse(args[2]) : _timeout;

        Module<Tensor, Tensor> model = IncResNet.IncResNetv1(device, weights);

        if (training) {
            Training(device, model, epochs, timeout);

            File.AppendAllText(_trainOutput, _trainStrings.ToString());
            File.AppendAllText(_validationOutput, _validationStrings.ToString());

            if (weights != null) {
                model.save(weights);
            } else {
                model.save(_weightsSavePath);
            }
        } else {
            _testStrings.AppendLine("target;prediction");
            Testing(device, model);
            File.WriteAllText(_testOutput, _testStrings.ToString());
        }
        model.Dispose();
    }
    /// <summary>
    /// Trains the model with the specified parameters.
    /// </summary>
    /// <param name="model">The model to be trained.</param>
    /// <param name="optimizer">The optimizer to use for training.</param>
    /// <param name="loss">The loss function to use for training.</param>
    /// <param name="dataLoader">The data loader for the training data.</param>
    /// <param name="epoch">The current epoch number.</param>
    /// <param name="size">The total number of samples in the training dataset.</param>
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
                float percent = (float)correct / total;
                _trainStrings.AppendLine($"{lossSq:0.000000};{percent:0.000000}");
                Console.WriteLine($"\r[Train] Epoch: {epoch} [{total} / {size}] MSE Root: {lossSq:0.000000} | " +
                    $"Total perfect prediction: {percent:0.000000}");
            }
            batchId++;

            d.DisposeEverything();
        }
    }
    /// <summary>
    /// Validates the model on the validation dataset.
    /// </summary>
    /// <param name="model">The model to be validated.</param>
    /// <param name="loss">The loss function to use for validation.</param>
    /// <param name="dataLoader">The data loader for the validation data.</param>
    /// <param name="size">The total number of samples in the validation dataset.</param>
    private static void Validate(
        Module<Tensor, Tensor> model,
        Loss<Tensor, Tensor, Tensor> loss,
        DataLoader dataLoader,
        long size) {

        model.eval();

        double lossSum = 0;
        long correct = 0;
        int batchCount = 0;

        using var d = NewDisposeScope();
        foreach (var data in dataLoader) {

            var target = data["label"];
            var prediction = model.call(data["data"]);
            var output = loss.call(prediction, target);

            lossSum += output.ToSingle();
            batchCount += 1;

            correct += prediction.to_type(int32).eq_(target.to_type(int32)).sum().ToInt32();

            d.DisposeEverything();
        }
        double lossSq = Math.Sqrt(lossSum) / batchCount;
        float percent = (float)correct / size;
        _validationStrings.AppendLine($"{lossSq:0.000000};{percent:0.000000}");
        Console.WriteLine($"\r[Validate] Average MSE Root: {lossSq:0.000000} | " +
            $"Total perfect prediction: {percent:0.000000}");
    }
    /// <summary>
    /// Tests the model on the test dataset.
    /// </summary>
    /// <param name="model">The model to be tested.</param>
    /// <param name="dataLoader">The data loader for the test data.</param>
    /// <param name="size">The total number of samples in the test dataset.</param>
    private static void Test(
    Module<Tensor, Tensor> model,
    DataLoader dataLoader,
    long size) {

        model.eval();

        long total = 0;

        using var d = NewDisposeScope();
        foreach (var data in dataLoader) {

            var target = data["label"];
            var prediction = model.call(data["data"]);

            long n = target.shape[0];
            for (int i = 0; i < n; i++) {
                _testStrings.AppendLine($"{(int)target[i]};{(float)prediction[i]}");
            }

            total += n;
            Console.WriteLine($"\r[Test]: {total} / {size}");

            d.DisposeEverything();
        }
    }
    /// <summary>
    /// Manages the training process of the model.
    /// </summary>
    /// <param name="device">The device to run the model on (e.g., CPU or CUDA).</param>
    /// <param name="model">The model to be trained.</param>
    /// <param name="epochs">The number of epochs to train for.</param>
    /// <param name="timeout">The maximum training time allowed.</param>
    private static void Training(Device device, Module<Tensor, Tensor> model, int epochs, int timeout) {
        Console.WriteLine($"\tRunning IncResNetv1 on {device.type} for {epochs} epochs, " +
            $"terminating after {TimeSpan.FromSeconds(timeout)}.");

        Console.WriteLine($"\tPreparing training and validation data...");

        using var trainData = new ImageDataset(_trainLoadPath);
        Console.WriteLine($"\tLoaded: {_trainLoadPath}");
        using var validationData = new ImageDataset(_validationLoadPath);
        Console.WriteLine($"\tLoaded: {_validationLoadPath}");

        using var train = DataLoader(trainData, _trainBatchSize, true, device);
        using var test = DataLoader(validationData, _otherBatchSize, false, device);

        using var optimizer = optim.Adam(model.parameters());

        var totalSW = new Stopwatch();
        totalSW.Start();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            var epochSW = new Stopwatch();
            epochSW.Start();

            _trainStrings.Append($"epoch-{epoch}" + Environment.NewLine);
            Train(model, optimizer, MSELoss(), train, epoch, trainData.Count);
            _validationStrings.Append($"epoch-{epoch}" + Environment.NewLine);
            Validate(model, MSELoss(), test, validationData.Count);

            epochSW.Stop();
            Console.WriteLine($"Elapsed time for this epoch: {epochSW.Elapsed.TotalSeconds}s.");

            if (totalSW.Elapsed.TotalSeconds > timeout) { break; }
        }
        totalSW.Stop();
        Console.WriteLine($"Elapsed training time: {totalSW.Elapsed}s.");
    }
    /// <summary>
    /// Manages the testing process of the model.
    /// </summary>
    /// <param name="device">The device to run the model on (e.g., CPU or CUDA).</param>
    /// <param name="model">The model to be tested.</param>
    private static void Testing(Device device, Module<Tensor, Tensor> model) {
        Console.WriteLine($"\tTesting IncResNetv1 on {device.type}.");
        Console.WriteLine($"\tPreparing test data...");

        using var testData = new ImageDataset(_testLoadPath);

        Console.WriteLine($"\tLoaded: {_testLoadPath}");

        using var test = DataLoader(testData, _otherBatchSize, false, device);

        var totalSW = new Stopwatch();
        totalSW.Start();

        Test(model, test, testData.Count);

        totalSW.Stop();
        Console.WriteLine($"Elapsed testing time: {totalSW.Elapsed}s.");
    }
}