using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision.io;

namespace IncResNet;
/// <summary>
/// Custom age labeled image loading dataset
/// </summary>
class ImageDataset : Dataset {
    private readonly string _rootDir;
    private readonly List<(string, int)> _images;

    public override long Count => _images.Count;

    public ImageDataset(string rootDir) {
        DefaultImager = new SkiaImager();

        _rootDir = rootDir;
        _images = LoadImages();
    }

    private List<(string, int)> LoadImages() {
        var imagesList = new List<(string, int)>();

        string[] classPaths = Directory.GetDirectories(_rootDir);

        for (int idx = 0; idx < classPaths.Length; idx++) {
            string classPath = classPaths[idx];
            string label = Path.GetFileName(classPath);

            string[] imageFiles = Directory.GetFiles(classPath, "*.jpg");

            foreach (string imgPath in imageFiles) {
                imagesList.Add((imgPath, int.Parse(label)));
            }
        }
        return imagesList;
    }

    public override Dictionary<string, Tensor> GetTensor(long index) {
        var (imgPath, label) = _images[(int)index];

        Tensor imageTensor = read_image(imgPath, ImageReadMode.RGB);
        imageTensor = imageTensor.to_type(float32);

        Tensor labelTensor = (float)label;

        return new Dictionary<string, Tensor>
        {
            { "data", imageTensor },
            { "label", labelTensor }
        };
    }
}
