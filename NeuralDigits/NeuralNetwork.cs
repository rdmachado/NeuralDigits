using Accord.Math.Optimization;
using System;
using System.Diagnostics;
using System.Linq;

namespace NeuralDigits
{
    class NeuralNetwork
    {
        // Neurons
        int input_layer,
            hidden_layer,
            output_layer;

        // Learning Parameters
        const double learning_rate = 0.1,
                     epsilon = 0.12;

        // Working values
        Matrix theta_1,
               theta_2,
               training_features,
               training_classes;
        
        public event EventHandler<OptimizationProgressEventArgs> OnBackPropagationProgress;

        public NeuralNetwork(int input_layer, int hidden_layer, int output_layer)
        {
            this.input_layer = input_layer;
            this.hidden_layer = hidden_layer;
            this.output_layer = output_layer;

            theta_1 = new Matrix(input_layer + 1, hidden_layer);
            theta_2 = new Matrix(hidden_layer + 1, output_layer);
            training_classes = new Matrix(1, output_layer);
        }

        #region Public Methods
        
        public void SetWeights(double[] weights)
        {
            theta_1 = Matrix.FromDoubleArray(weights.Take((input_layer + 1) * hidden_layer).ToArray(), hidden_layer);
            theta_2 = Matrix.FromDoubleArray(weights.Skip((input_layer + 1) * hidden_layer).ToArray(), output_layer);
        }

        public double[] GetWeights()
        {
            return AppendArrays(theta_1.ToArray(), theta_2.ToArray());
        }

        public double[] FeedForward(double[] features)
        {
            Matrix input = Matrix.FromDoubleArray(features, features.Length).AddBiasUnit();

            // calculate output layer activation values for the given features
            double[] outputLayerActivation = ((input * theta_1).ApplyFunction(Sigmoid).AddBiasUnit() * theta_2).ApplyFunction(Sigmoid).ToArray();

            return outputLayerActivation;
        }
        
        public void TrainBackPropagation(double[] features, int[] classes, int iterations)
        {
            training_features = Matrix.FromDoubleArray(features, input_layer);
            training_classes = Matrix.Unroll(classes, output_layer);
            
            ConjugateGradient cg = new ConjugateGradient(
                ((input_layer + 1) * hidden_layer) + ((hidden_layer + 1) * output_layer),
                CostFunction, Gradient);

            cg.MaxIterations = iterations;
            cg.Progress += ConjugateDescentProgress;
            cg.Minimize();
            double[] solution = cg.Solution;

            theta_1 = Matrix.FromDoubleArray(solution.Take((input_layer + 1) * hidden_layer).ToArray(), hidden_layer);
            theta_2 = Matrix.FromDoubleArray(solution.Skip((input_layer + 1) * hidden_layer).ToArray(), output_layer);
        }

        #endregion

        #region Inner Workings

        private void ConjugateDescentProgress(object sender, OptimizationProgressEventArgs e)
        {
            Debug.WriteLine("Iteration: " + e.Iteration + ", Current cost: " + e.Value);
            OnBackPropagationProgress?.Invoke(this, e);
        }

        private double CostFunction(double[] weights)
        {
            double cost = 0;

            Matrix theta_1 = Matrix.FromDoubleArray(weights.Take((input_layer + 1) * hidden_layer).ToArray(), hidden_layer),
                   theta_2 = Matrix.FromDoubleArray(weights.Skip((input_layer + 1) * hidden_layer).ToArray(), output_layer);

            Matrix hTheta = ((training_features.AddBiasUnit() * theta_1).ApplyFunction(Sigmoid).AddBiasUnit() * theta_2).ApplyFunction(Sigmoid);

            // unregularized cost
            Matrix leftMember = (0 - training_classes).ElementWiseMultiplication(hTheta.ApplyFunction(Math.Log)),
                   rightMember = (1 - training_classes).ElementWiseMultiplication((1 - hTheta).ApplyFunction(Math.Log));

            cost = (leftMember - rightMember).CollapseSum() / training_features.Rows;

            // regularization member
            double regularizationLeftMember = theta_1.Transpose().RemoveBiasUnit().ApplyFunction(n => Math.Pow(n, 2)).CollapseSum();
            double regularizationRightMember = theta_2.Transpose().RemoveBiasUnit().ApplyFunction(n => Math.Pow(n, 2)).CollapseSum();
            
            cost += (regularizationLeftMember + regularizationRightMember) * (learning_rate / (2 * training_features.Rows));

            //Debug.WriteLine("Current cost: " + cost);

            return cost;
        }

        private double[] Gradient(double[] weights)
        {
            Matrix theta_1 = Matrix.FromDoubleArray(weights.Take((input_layer + 1) * hidden_layer).ToArray(), hidden_layer),
                   theta_2 = Matrix.FromDoubleArray(weights.Skip((input_layer + 1) * hidden_layer).ToArray(), output_layer);

            Matrix delta1 = new Matrix(theta_1.Columns, theta_1.Rows),
                   delta2 = new Matrix(theta_2.Columns, theta_2.Rows);

            for (int i = 0; i < training_features.Rows; i++)
            {
                Matrix a1 = training_features.GetRow(i).AddBiasUnit(),
                       z2 = a1 * theta_1,
                       a2 = z2.ApplyFunction(Sigmoid).AddBiasUnit(),
                       z3 = a2 * theta_2,
                       a3 = z3.ApplyFunction(Sigmoid),
                       d3 = a3 - training_classes.GetRow(i),
                       d2 = (d3 * theta_2.Transpose().RemoveBiasUnit()).ElementWiseMultiplication(z2.ApplyFunction(SigmoidGradient));
                
                delta1 += (d2.Transpose() * a1);
                delta2 += (d3.Transpose() * a2);
            }

            double[] theta_1_gradient = ((delta1 / training_features.Rows) + (theta_1.Transpose().SetColumn(0, 0) * (learning_rate / training_features.Rows))).Transpose().ToArray(),
                     theta_2_gradient = ((delta2 / training_features.Rows) + (theta_2.Transpose().SetColumn(0, 0) * (learning_rate / training_features.Rows))).Transpose().ToArray();
            
            return AppendArrays(theta_1_gradient, theta_2_gradient);
        }

        #endregion

        #region Auxiliary Methods

        private double[] AppendArrays(double[] array_1, double[] array_2)
        {
            var t1arr = array_1.ToArray();
            var t2arr = array_2.ToArray();

            double[] ret = new double[t1arr.Length + t2arr.Length];

            Array.Copy(t1arr, ret, t1arr.Length);
            Array.Copy(t2arr, 0, ret, t1arr.Length, t2arr.Length);

            return ret;
        }

        private double Sigmoid(double z)
        {
            return 1 / (1 + Math.Exp(-z));
        }

        private double SigmoidGradient(double z)
        {
            return Sigmoid(z) * (1 - Sigmoid(z));
        }

        #endregion
    }
}
