using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision.io;

namespace IncResNet;
/// <summary>
/// Custom dataset class for loading images and their associated labels (ages), and serving them as tensors.
/// </summary>
class ImageDataset : Dataset {
    private readonly string _rootDir;
    private readonly List<(string, int)> _images;
    /// <summary>
    /// Gets the number of images in the dataset.
    /// </summary>
    public override long Count => _images.Count;
    /// <summary>
    /// Initializes a new instance of the <see cref="ImageDataset"/> class.
    /// </summary>
    /// <param name="rootDir">The root directory containing the image folders.</param>
    public ImageDataset(string rootDir) {
        DefaultImager = new SkiaImager();

        _rootDir = rootDir;
        _images = LoadImages();
    }
    /// <summary>
    /// Loads the images from the root directory and creates a list of image paths and their corresponding labels.
    /// </summary>
    /// <returns>A list of tuples where each tuple contains the image path and its label.</returns>
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
    /// <summary>
    /// Gets the tensor representation of the image and its label at the specified index.
    /// </summary>
    /// <param name="index">The index of the image and label to retrieve.</param>
    /// <returns>A dictionary containing the image tensor with key "data" and the label tensor with key "label".</returns>
    public override Dictionary<string, Tensor> GetTensor(long index) {
        var (imgPath, label) = _images[(int)index];

        Tensor imageTensor = read_image(imgPath, ImageReadMode.RGB);
        imageTensor = imageTensor.to_type(float32);

        Tensor labelTensor = new float[] { label };

        return new Dictionary<string, Tensor>
        {
            { "data", imageTensor },
            { "label", labelTensor }
        };
    }
}
