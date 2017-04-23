using System;
using System.Threading.Tasks;

namespace NeuralDigits
{
    class Matrix
    {
        public readonly int Rows, Columns;
        public double[,] matrix;
        public Matrix(int Rows, int Columns)
        {
            this.Rows = Rows; this.Columns = Columns;
            matrix = new double[Rows, Columns];
        }

        public Matrix(int size, bool identity)
        {
            this.Rows = size; this.Columns = size;
            matrix = new double[Rows, Columns];
            if (identity)
            {
                for (int i = 0; i < size; i++)
                {
                    for (int j = 0; j < size; j++)
                    {
                        if (i == j)
                            matrix[i, j] = 1;
                    }
                }
            }
        }

        public double this[int Row, int Column]
        {
            get { return matrix[Row, Column]; }
            set { matrix[Row, Column] = value; }            
        }

        #region Operators

        public static Matrix operator +(Matrix matrix_1, Matrix matrix_2)
        {
            Matrix ret = new Matrix(matrix_1.Rows, matrix_1.Columns);
            for (int i = 0; i < matrix_1.Rows; i++)
            {
                for (int j = 0; j < matrix_1.Columns; j++)
                {
                    ret[i, j] = matrix_1[i, j] + matrix_2[i, j];
                }
            }
            return ret;
        }

        public static Matrix operator +(Matrix matrix_1, double value)
        {
            return matrix_1.ApplyFunction(n => n + value);
        }

        public static Matrix operator +(double value, Matrix matrix_1)
        {
            return matrix_1.ApplyFunction(n => value + n);
        }

        public static Matrix operator -(Matrix matrix_1, Matrix matrix_2)
        {
            Matrix ret = new Matrix(matrix_1.Rows, matrix_1.Columns);
            for (int i = 0; i < matrix_1.Rows; i++)
            {
                for (int j = 0; j < matrix_1.Columns; j++)
                {
                    ret[i, j] = matrix_1[i, j] - matrix_2[i, j];
                }
            }
            return ret;
        }

        public static Matrix operator -(Matrix matrix_1, double value)
        {
            return matrix_1.ApplyFunction(n => n - value);
        }

        public static Matrix operator -(double value, Matrix matrix_1)
        {
            return matrix_1.ApplyFunction(n => value - n);
        }

        public static Matrix operator *(Matrix matrix_1, Matrix matrix_2)
        {
            Matrix ret = new Matrix(matrix_1.Rows, matrix_2.Columns);

            Parallel.For(0, matrix_1.Rows, i =>
            {
                for (int j = 0; j < matrix_2.Columns; j++)
                {
                    for (int k = 0; k < matrix_1.Columns; k++)
                    {
                        ret[i, j] += matrix_1[i, k] * matrix_2[k, j];
                    }
                }
            });

            return ret;
        }

        public static Matrix operator *(Matrix matrix_1, double value)
        {
            return matrix_1.ApplyFunction(n => n * value);
        }

        public static Matrix operator *(double value, Matrix matrix_1)
        {
            return matrix_1.ApplyFunction(n => n * value);
        }

        public static Matrix operator /(Matrix matrix_1, Matrix matrix_2)
        {
            Matrix ret = new Matrix(matrix_1.Rows, matrix_1.Columns);
            for (int i = 0; i < matrix_1.Rows; i++)
            {
                for (int j = 0; j < matrix_1.Columns; j++)
                {
                    ret[i, j] = matrix_1[i, j] / matrix_2[i, j];
                }
            }
            return ret;
        }

        public static Matrix operator /(Matrix matrix_1, double value)
        {
            return matrix_1.ApplyFunction(n => n / value);
        }

        public static Matrix operator /(double value, Matrix matrix_1)
        {
            return matrix_1.ApplyFunction(n => value / n);
        }

        public static bool operator ==(Matrix matrix_1, Matrix matrix_2)
        {
            if (matrix_1.Rows != matrix_2.Rows || matrix_1.Columns != matrix_2.Columns)
                return false;
            else
            {
                for (int i = 0; i < matrix_1.Rows; i++)
                {
                    for(int j = 0; j < matrix_2.Columns; j++)
                    {
                        if (matrix_1[i, j] != matrix_2[i, j])
                            return false;
                    }
                }
            }
            return true;
        }

        public static bool operator !=(Matrix matrix_1, Matrix matrix_2)
        {
            if (matrix_1.Rows != matrix_2.Rows || matrix_1.Columns != matrix_2.Columns)
                return true;
            else
            {
                for (int i = 0; i < matrix_1.Rows; i++)
                {
                    for (int j = 0; j < matrix_2.Columns; j++)
                    {
                        if (matrix_1[i, j] != matrix_2[i, j])
                            return true;
                    }
                }
            }
            return false;
        }

        #endregion
        
        public static Matrix FromDoubleArray(double[] array, int rowSize)
        {
            Matrix ret = new Matrix(array.Length / rowSize, rowSize);
            int ind = 0;
            for (int i = 0; i < array.Length / rowSize; i++)
            {
                for (int j = 0; j < rowSize; j++)
                {
                    ret[i, j] = array[ind];
                    ind++;
                }
            }
            return ret;
        }
        
        public static Matrix Unroll(int[] input, int classes)
        {
            Matrix eye = new Matrix(classes, true);
            Matrix ret = new Matrix(input.Length, classes);
            for (int i = 0; i < input.Length; i++)
            {
                ret[i, input[i]] = 1;
            }
            return ret;
        }

        public Matrix Transpose()
        {
            Matrix ret = new Matrix(Columns, Rows);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    ret[j, i] = this[i, j];
                }
            }
            return ret;
        }
        
        public Matrix GetColumn(int index)
        {
            Matrix ret = new Matrix(Rows, 1);
            for (int i = 0; i < Rows; i++)
            {
                ret[i, 0] = this[i, index];
            }
            return ret;
        }

        public Matrix GetRow(int index)
        {
            Matrix ret = new Matrix(1, Columns);
            for (int i = 0; i < Columns; i++)
            {
                ret[0, i] = this[index, i];
            }
            return ret;
        }

        public Matrix SetColumn(int index, double value)
        {
            Matrix ret = this;
            for (int i = 0; i < Rows; i++)
            {
                ret[i, index] = value;
            }
            return ret;
        }

        public Matrix SetRow(int index, double value)
        {
            Matrix ret = this;
            for (int i = 0; i < Columns; i++)
            {
                ret[index, i] = value;
            }
            return ret;
        }
        
        public Matrix ElementWiseMultiplication(Matrix matrix2)
        {
            Matrix ret = new Matrix(this.Rows, this.Columns);
            for (int i = 0; i < this.Rows; i++)
            {
                for (int j = 0; j < this.Columns; j++)
                {
                    ret[i, j] = this[i, j] * matrix2[i, j];
                }
            }
            return ret;
        }

        public Matrix AddColumn(bool preappend)
        {
            Matrix ret = new Matrix(Rows, Columns + 1);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns + 1; j++)
                {
                    if (preappend)
                        ret[i, j] = j == 0 ? 0 : this[i, j - 1];
                    else
                        ret[i, j] = j == Columns ? 0 : this[i, j];
                }
            }
            return ret;
        }

        public Matrix AddRow(bool preappend)
        {
            int index = preappend ? 0 : Rows;
            Matrix ret = new Matrix(Rows + 1, Columns);
            for (int i = 0; i < Rows + 1; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    if (preappend)
                        ret[i, j] = i == 0 ? 0 : this[i - 1, j];
                    else
                        ret[i, j] = i == Rows ? 0 : this[i, j];
                }
            }
            return ret;
        }

        public Matrix AddBiasUnit()
        {
            Matrix ret = new Matrix(Rows, Columns + 1);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns + 1; j++)
                {
                    ret[i, j] = j == 0 ? 1 : this[i, j - 1];
                }
            }
            return ret;
        }

        public Matrix RemoveBiasUnit()
        {
            Matrix ret = new Matrix(Rows, Columns - 1);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 1; j < Columns; j++)
                {
                    ret[i, j - 1] = this[i, j];
                }
            }
            return ret;
        }

        public Matrix ApplyFunction(Func<double, double> function)
        {
            Matrix ret = new Matrix(Rows, Columns);

            Parallel.For(0, Rows, i =>
            {
                for (int j = 0; j < Columns; j++)
                {
                    ret[i, j] = function(this[i, j]);
                }
            });

            return ret;
        }
        
        public double CollapseSum()
        {
            double ret = 0;
            foreach (double d in this.matrix)
            {
                ret += d;
            }
            return ret;
        }

        public double[] ToArray()
        {
            double[] ret = new double[Rows * Columns];
            int k = 0;
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    ret[k] = this[i, j];
                    k++;
                }
            }
            return ret;
        }

        public static Matrix CreateRandomized(int Rows, int Columns)
        {
            Matrix ret = new Matrix(Rows, Columns);
            Random rand = new Random();
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    ret[i, j] = rand.NextDouble();
                }
            }
            return ret;
        }

        public static Matrix CreateSinomized(int Rows, int Columns)
        {
            Matrix ret = new Matrix(Rows, Columns);
            int n = 1;
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    ret[i, j] = Math.Sin(n) / 10;
                    n++;
                }
            }
            return ret;
        }
    }
}
