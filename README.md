# IncResNet
IncResNet, or in full words Inceptive Residual Network, combines the ideas presented in GoogLeNet (*Going deeper with convolutions*) and ResNet (*Deep residual learning for image recognition*) papers. Many solutions used for the architecture were also inspired by Inception-ResNet paper (*Inception-v4, inception-resnet and the impact of
residual connections on learning*), which already discussed the proposed idea, but varies in implementation.

The model was implemented in TorchSharp on .NET 7.

## Command line arguments
```
<weights>
The path to the .bin file with serialized weights.

<epochs>
The number of epochs to perform.

<timeout>
The maximum learning time for the session in hours.
```

## Constants
What cannot be set from CLI, must be set from code. At the beginning of the `Configuration.cs` class you can find:
```csharp
private static int _trainBatchSize = 32;
private static int _otherBatchSize = 64;
private static int _epochs = 5;

private static readonly string _weightsSavePath = @".\weights.bin";
private static readonly string _trainLoadPath = @".\dataset\training";
private static readonly string _validationLoadPath = @".\dataset\validation";
private static readonly string _testLoadPath = @".\dataset\test";

private static readonly int _loggingInterval = 20;
private static readonly int _timeout = 2 * 3600;

private static readonly string _trainOutput = @".\train.txt";
private static readonly string _validationOutput = @".\validation.txt";
private static readonly string _testOutput = @".\test.txt";
```

In `Program.cs` a function accepting a boolean is called:
```csharp
Configuration.Start(args, false);
```
If you want to train the model, pass `true` as second argument, otherwise, if you want to just test it, pass `false`.

Technically, all of this should be set in some configuration file - I will consider it in the future.

## Dependencies
Program uses a few TorchSharp libraries in a form of NuGet packages:
- TorchSharp
- TorchVision

and a suitable backend for TorchSharp (choose CUDA if you have Nvidia GPU, otherwise CPU).

## Contact
If you have any questions or suggestions feel free to contact me at ordinary.email.address@protonmail.com.

---
*Copyright © 2024 Bartosz Kaczorowski*
