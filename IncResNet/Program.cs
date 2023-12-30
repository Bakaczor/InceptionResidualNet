using TorchSharp;
using static TorchSharp.torch;

namespace IncResNet; 
internal class Program {
    static void Main(string[] args) {
        var device = new Device(DeviceType.CUDA);
        var net = IncResNet.IncResNetv1(768, device);
        var y = net.forward(randn(1, 3, 200, 200).cuda(device));
        Console.WriteLine(y);
    }
}
