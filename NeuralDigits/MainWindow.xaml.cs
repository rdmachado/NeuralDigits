using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Ink;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace NeuralDigits
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        DigitRecognizer dr;
        public MainWindow()
        {
            InitializeComponent();
            dr = new DigitRecognizer();
            dr.LoadWeights(@"./PreCompTheta.dat"); // Set the default neural network weights
        }

        private void Guess()
        {
            var processedImage = PreProcessCanvasImage();

            if (processedImage == null)
                labelGuess.Content = "";
            else
            {
                Clipboard.SetImage(BitmapFrame.Create(processedImage.Item1));

                var result = dr.Predict(processedImage.Item2);

                labelGuess.Content = result.Item2;
            }
        }

        // Load the training set and train the network
        private void Train()
        {
            byte[,] images;
            byte[] labels;
            ReadMNIST(@"./train-images.idx3-ubyte", @"./train-labels.idx1-ubyte", 5000, out images, out labels);

            dr.TrainProgressChanged += DigitRecognizer_TrainProgressChanged;
            dr.TrainComplete += DigitRecognizer_TrainComplete;

            ToggleNNControls(false);

            Task task = new Task(() => dr.Learn(images, labels, 10));
            task.Start();
        }

        // Load the test set and test the network
        private void Test()
        {
            byte[,] images;
            byte[] labels;
            ReadMNIST(@"./t10k-images.idx3-ubyte", @"./t10k-labels.idx1-ubyte", 10000, out images, out labels);

            dr.TestProgressChanged += DigitRecognizer_TestProgressChanged;
            dr.TestComplete += DigitRecognizer_TestComplete;

            ToggleNNControls(false);

            Thread th = new Thread(() => dr.Test(images, labels));
            th.Start();
        }

        // Render the contents of the InkCanvas and preprocess the resulting image to better resemble the MNIST database items
        private Tuple<BitmapSource, byte[]> PreProcessCanvasImage()
        {
            if (canvasDigitBoard.Strokes.Count() == 0) return null;

            canvasDigitBoard.Select(canvasDigitBoard.Strokes);

            int boundLeft = (int)canvasDigitBoard.GetSelectionBounds().Left < 0 ? 0 : (int)canvasDigitBoard.GetSelectionBounds().Left,
                boundTop = (int)canvasDigitBoard.GetSelectionBounds().Top < 0 ? 0 : (int)canvasDigitBoard.GetSelectionBounds().Top,
                boundRight = (int)canvasDigitBoard.GetSelectionBounds().Right > (int)canvasDigitBoard.ActualWidth ? (int)canvasDigitBoard.ActualWidth : (int)canvasDigitBoard.GetSelectionBounds().Right,
                boundBottom = (int)canvasDigitBoard.GetSelectionBounds().Bottom > (int)canvasDigitBoard.ActualHeight ? (int)canvasDigitBoard.ActualHeight : (int)canvasDigitBoard.GetSelectionBounds().Bottom;                

            // render the contents of the InkCanvas to a bitmap
            int margin = (int)canvasDigitBoard.Margin.Left;
            RenderTargetBitmap bmp = new RenderTargetBitmap(
                (int)canvasDigitBoard.ActualWidth - margin,
                (int)canvasDigitBoard.ActualHeight - margin,
                0, 0, PixelFormats.Default);
            bmp.Render(canvasDigitBoard);

            // trim the background by cropping only the content
            CroppedBitmap cropbmp = new CroppedBitmap(bmp, new Int32Rect(boundLeft, boundTop, boundRight - boundLeft, boundBottom - boundTop));

            // scale to fit in a 20x20 pixel box maintaining aspect ratio
            double scaleFactor = (double)20 / Math.Max(cropbmp.PixelHeight, cropbmp.PixelWidth);
            ScaleTransform scaleTransform = new ScaleTransform(scaleFactor, scaleFactor);
            TransformedBitmap scalebmp = new TransformedBitmap(cropbmp, scaleTransform);

            // convert bitmap to 8-bit grayscale
            FormatConvertedBitmap convbmp = new FormatConvertedBitmap();
            convbmp.BeginInit();
            convbmp.Source = scalebmp;
            convbmp.DestinationFormat = PixelFormats.Gray8;
            convbmp.EndInit();

            // center the bitmap by its center of mass point and pad it to its final size
            var paddedbmp = PadBitmapFromCenter(convbmp, GetBitmapCenterOfMass(convbmp), 28, 28);

            canvasDigitBoard.Select(new StrokeCollection());

            canvasDigitBoard.EditingMode = System.Windows.Controls.InkCanvasEditingMode.Ink;

            return paddedbmp;
        }

        // Calculate a bitmap's center of mass
        private Tuple<int, int> GetBitmapCenterOfMass(BitmapSource image)
        {
            byte[] buffer = new byte[image.PixelHeight * image.PixelWidth];
            int stride = (image.PixelWidth * image.Format.BitsPerPixel + 7) / 8;

            image.CopyPixels(buffer, stride, 0);

            double horizontalCentroid = 0,
                   verticalCentroid = 0,
                   totalValue = 0;
            for (int i = 0; i < image.PixelHeight; i++)
            {
                for (int j = 0; j < image.PixelWidth; j++)
                {
                    int pixel = i * image.PixelWidth + j;

                    verticalCentroid += (i+1) * buffer[pixel];
                    horizontalCentroid += (j+1) * buffer[pixel];
                    totalValue += buffer[pixel];
                }
            }

            horizontalCentroid /= totalValue;
            verticalCentroid /= totalValue;

            return new Tuple<int, int>((int)Math.Round(horizontalCentroid), (int)Math.Round(verticalCentroid));
        }

        // Create a bitmap with the specified size, with the original bitmap centered on a custom point
        private Tuple<BitmapSource, byte[]> PadBitmapFromCenter(BitmapSource image, Tuple<int, int> centerPoint, int targetPixelWidth, int targetPixelHeight)
        {
            byte[] buffer = new byte[image.PixelHeight * image.PixelWidth];
            int stride = (image.PixelWidth * image.Format.BitsPerPixel + 7) / 8;
            image.CopyPixels(buffer, stride, 0);

            Matrix mat = Matrix.FromDoubleArray(buffer.Select(e => (double)e).ToArray(), image.PixelWidth);

            int columnstoadd = targetPixelWidth - image.PixelWidth,
                rowstoadd = targetPixelHeight - image.PixelHeight,
                relCenterX = centerPoint.Item1, relCenterY = centerPoint.Item2;

            while (mat.Rows < targetPixelHeight || mat.Columns < targetPixelWidth)
            {
                if (mat.Columns < targetPixelWidth)
                {
                    if (relCenterX <= (int)(mat.Columns / 2))
                    {
                        mat = mat.AddColumn(true);
                        relCenterX++;
                    }
                    else
                    {
                        mat = mat.AddColumn(false);
                    }
                }

                if (mat.Rows < targetPixelHeight)
                {
                    if (relCenterY <= (int)(mat.Rows / 2))
                    {
                        mat = mat.AddRow(true);
                        relCenterY++;
                    }
                    else
                    {
                        mat = mat.AddRow(false);
                    }
                }
            }

            byte[] finalImgBuffer = mat.ToArray().Select(e => (byte)e).ToArray();
            stride = (mat.Columns * image.Format.BitsPerPixel + 7) / 8;

            var ret = BitmapSource.Create(mat.Columns, mat.Rows, 0, 0, PixelFormats.Gray8, null, finalImgBuffer, stride);

            return new Tuple<BitmapSource, byte[]>(ret, finalImgBuffer);
        }
        
        // Read the database as per the format specified at http://yann.lecun.com/exdb/mnist/
        private void ReadMNIST(string imagesFilePath, string labelsFilePath, int readCount, out byte[,] images, out byte[] labels)
        {
            using (FileStream ifs = new FileStream(imagesFilePath, FileMode.Open))
            {
                using (FileStream lfs = new FileStream(labelsFilePath, FileMode.Open))
                {
                    BinaryReader imagesReader = new BinaryReader(ifs);
                    BinaryReader labelsReader = new BinaryReader(lfs);

                    byte[] intBuffer = imagesReader.ReadBytes(4);
                    int magicNumber = BitConverter.ToInt32(intBuffer.Reverse().ToArray(), 0);

                    intBuffer = imagesReader.ReadBytes(4);
                    int imageCount = BitConverter.ToInt32(intBuffer.Reverse().ToArray(), 0);

                    intBuffer = imagesReader.ReadBytes(4);
                    int rows = BitConverter.ToInt32(intBuffer.Reverse().ToArray(), 0);

                    intBuffer = imagesReader.ReadBytes(4);
                    int columns = BitConverter.ToInt32(intBuffer.Reverse().ToArray(), 0);

                    if (readCount > 0 && readCount < imageCount)
                        imageCount = readCount;

                    byte[,] imagesBuffer = new byte[imageCount, rows * columns];

                    for (int i = 0; i < imageCount; i++)
                    {
                        for (int j = 0; j < rows * columns; j++)
                        {
                            imagesBuffer[i, j] = imagesReader.ReadByte();
                        }
                    }

                    intBuffer = labelsReader.ReadBytes(4);
                    magicNumber = BitConverter.ToInt32(intBuffer.Reverse().ToArray(), 0);

                    intBuffer = labelsReader.ReadBytes(4);
                    imageCount = BitConverter.ToInt32(intBuffer.Reverse().ToArray(), 0);

                    if (readCount > 0 && readCount < imageCount)
                        imageCount = readCount;

                    byte[] labelsBuffer = new byte[imageCount];

                    for (int i = 0; i < imageCount; i++)
                    {
                        labelsBuffer[i] = labelsReader.ReadByte();
                    }

                    images = imagesBuffer;
                    labels = labelsBuffer;
                }
            }
        }
        
        private void ToggleNNControls(bool enable)
        {
            btnTest.IsEnabled = enable;
            btnTrain.IsEnabled = enable;
            btnSaveWeights.IsEnabled = enable;
            btnResetWeights.IsEnabled = enable;
        }

        #region Event Handlers

        private void btnSaveWeights_Click(object sender, RoutedEventArgs e)
        {
            dr.SaveWeights(@"./UserTheta.dat");
        }

        private void canvasDigitBoard_StrokeCollected(object sender, System.Windows.Controls.InkCanvasStrokeCollectedEventArgs e)
        {
            Guess();
        }

        private void canvasDigitBoard_StrokeErased(object sender, RoutedEventArgs e)
        {
            Guess();
        }

        private void canvasDigitBoard_StrokesReplaced(object sender, System.Windows.Controls.InkCanvasStrokesReplacedEventArgs e)
        {
            Guess();
        }

        private void btnResetWeights_Click(object sender, RoutedEventArgs e)
        {
            dr.ResetNeuralNetwork(); // Reset the nnet in case it was using a reduced hidden layer size
            dr.LoadWeights(@"./pyweights.txt");
        }
        
        private void DigitRecognizer_TestProgressChanged(object source, DigitRecognizer.TestProgressChangedEventArgs e)
        {
            try
            {
                labelStatus.Dispatcher.Invoke(
                () => labelStatus.Content = "Test: " + e.testCount + " / 10000, Acc.: " + ((double)e.correctCount / e.testCount * 100).ToString("#.##")
            );
            }
            catch
            {
                Thread.CurrentThread.Abort();
            }
        }

        private void DigitRecognizer_TestComplete(object source, DigitRecognizer.TestProgressChangedEventArgs e)
        {
            try
            {
                this.Dispatcher.Invoke(
                    () => ToggleNNControls(true)
                );
            }
            catch
            {
                Thread.CurrentThread.Abort();
            }
        }

        private void DigitRecognizer_TrainProgressChanged(object source, DigitRecognizer.TrainProgressChangedEventArgs e)
        {
            try
            {
                labelStatus.Dispatcher.Invoke(
                () => labelStatus.Content = "Training iteration: " + (e.iteration > 0 ? e.iteration - 1 : e.iteration) + " / 10, Current cost: " + e.cost.ToString("#.##")
            );
            }
            catch
            {
                Thread.CurrentThread.Abort();
            }
        }

        private void DigitRecognizer_TrainComplete(object source, DigitRecognizer.TrainProgressChangedEventArgs e)
        {
            try
            {
                this.Dispatcher.Invoke(
                    () => ToggleNNControls(true)
                );
            }
            catch
            {
                Thread.CurrentThread.Abort();
            }
        }

        private void btnTrain_Click(object sender, RoutedEventArgs e)
        {
            Train();
        }

        private void btnTest_Click(object sender, RoutedEventArgs e)
        {
            Test();
        }
        
        private void btnClearCanvas_Click(object sender, RoutedEventArgs e)
        {
            canvasDigitBoard.Strokes.Clear();
        }

        #endregion
    }
}
