using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralDigits
{
    class DigitRecognizer
    {
        // Create a 3-layer neural network classifier with the specified number of neurons
        static NeuralNetwork nnet;

        public delegate void TestProgressChangedEventHandler(object source, TestProgressChangedEventArgs e);
        public delegate void TrainProgressChangedEventHandler(object source, TrainProgressChangedEventArgs e);
        public class TestProgressChangedEventArgs : EventArgs
        {
            public readonly int testCount,
                                correctCount;
            public TestProgressChangedEventArgs(int testCount, int correctCount)
            {
                this.testCount = testCount;
                this.correctCount = correctCount;
            }
        }

        public class TrainProgressChangedEventArgs : EventArgs
        {
            public readonly int iteration;
            public readonly double cost;
            public TrainProgressChangedEventArgs(int iteration, double cost)
            {
                this.iteration = iteration;
                this.cost = cost;
            }
        }

        public event TestProgressChangedEventHandler TestProgressChanged,
                                                     TestComplete;

        public event TrainProgressChangedEventHandler TrainProgressChanged,
                                                      TrainComplete;

        public DigitRecognizer()
        {
            nnet = new NeuralNetwork(784, 300, 10);
        }

        #region Public Methods

        public void ResetNeuralNetwork()
        {
            nnet = new NeuralNetwork(784, 300, 10);
        }

        public void LoadWeights(string filePath)
        {
            // read pre-calculated weights from file
            double[] weights = File.ReadAllLines(@filePath).Select(n => double.Parse(n)).ToArray();
            nnet.SetWeights(weights);
        }

        public void SaveWeights(string filePath)
        {
            File.WriteAllLines(@filePath, nnet.GetWeights().Select(d => d.ToString()).ToArray());
        }

        public Tuple<double, int> Predict(byte[] pixels)
        {
            // feed the neural network with normalized pixel values (double values ranging from 0 to 1)
            double[] results = nnet.FeedForward(pixels.Select(n => NormalizePixelValues(n)).ToArray());

            // pick the result with the highest chance of being the correct one
            double maxConfidence = 0;
            int digit = 0;
            for (int i = 0; i < results.Length; i++)
            {
                if (results[i] > maxConfidence)
                {
                    maxConfidence = results[i];
                    digit = i;
                }
            }

            return new Tuple<double, int>(maxConfidence, digit);
        }

        public void Learn(byte[,] input, byte[] input_tests, int iterations)
        {
            nnet = new NeuralNetwork(784, 5, 10); // Create a much smaller network for demonstration purposes

            double[] features = new double[input.GetLength(0) * input.GetLength(1)];
            int k = 0;
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    features[k] = NormalizePixelValues(input[i, j]);
                    k++;
                }
            }
            nnet.OnBackPropagationProgress += Nnet_OnBackPropagationProgress;

            nnet.TrainBackPropagation(features, input_tests.Select(n => (int)n).ToArray(), iterations);

            TrainComplete?.Invoke(this, new TrainProgressChangedEventArgs(iterations, 0));
        }

        public void Test(byte[,] input, byte[] input_tests)
        {
            List<double[]> features = new List<double[]>();
            for (int i = 0; i < input.GetLength(0); i++)
            {
                double[] f = new double[input.GetLength(1)];
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    f[j] = NormalizePixelValues(input[i, j]);
                }
                features.Add(f);
            }

            int count = 0, correct = 0;

            for (int i = 0; i < input.GetLength(0); i++)
            {
                var res = nnet.FeedForward(features[i]);
                count++;
                double maxConfidence = 0;
                int digit = 0;
                for (int j = 0; j < res.Length; j++)
                {
                    if (res[j] > maxConfidence)
                    {
                        maxConfidence = res[j];
                        digit = j;
                    }
                }
                if (digit == input_tests[i]) correct++;

                TestProgressChanged?.Invoke(this, new TestProgressChangedEventArgs(count, correct));

                //Debug.WriteLine("Predicted: " + digit + " Real: " + input_tests[i] + " Acc.: " + ((double)correct / count * 100).ToString("#.##") + "% Total: " + count + " Correct: " + correct);
            }

            TestComplete?.Invoke(this, new TestProgressChangedEventArgs(count, correct));
        }

        #endregion

        private void Nnet_OnBackPropagationProgress(object sender, Accord.Math.Optimization.OptimizationProgressEventArgs e)
        {
            TrainProgressChanged?.Invoke(this, new TrainProgressChangedEventArgs(e.Iteration, e.Value));
        }

        private double NormalizePixelValues(byte value)
        {
            return (double)value / 255;
        }

        // Debug function that creates a small neural network with predetermined weights
        private void TestNet()
        {
            nnet = new NeuralNetwork(2, 2, 4);

            double[] features = { Math.Cos(1), Math.Cos(2), Math.Cos(3), Math.Cos(4), Math.Cos(5), Math.Cos(6) };

            int[] classes = { 3, 1, 2 };

            double[] weights = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8 };


            nnet.SetWeights(weights);

            nnet.TrainBackPropagation(features, classes, 50);
        }

    }
}
