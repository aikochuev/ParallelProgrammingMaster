#include "stdafx.h"
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <omp.h>

struct CRSMatrix
{
    int n;
    int m; 
    int nz; 
    std::vector<double> val;
    std::vector<int> colIndex;
    std::vector<int> rowPtr;
};

void MultMatrixVector(CRSMatrix& A, double* b, double* x)
{
#pragma omp parallel for
    for(int i = 0; i < A.n; ++i)
    {
        x[i] = 0.;
        for(int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j)
            x[i] += A.val[j] * b[A.colIndex[j]];
    }
}
double MultVectorVector(int n, double* a, double* b)
{
    double sum = 0.;
#pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

void TransposeMatrix(CRSMatrix& A, CRSMatrix& At)
{
    At.n = A.m;
    At.m = A.n;
    At.nz = A.nz;
    At.rowPtr.resize(A.n + 1);
    At.colIndex.resize(A.nz);
    At.val.resize(A.nz);
    for(int i = 0; i < A.nz; i++)
        At.rowPtr[A.colIndex[i] + 1]++;
    int sum = 0;
    for(int i = 1; i <= A.n; i++)
    {
        int tmp = At.rowPtr[i];
        At.rowPtr[i] = sum;
        sum += tmp;
    }

    for(int i = 0; i < A.n; i++)
        for(int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; j++)
        {
            At.val[At.rowPtr[A.colIndex[j] + 1]] = A.val[j];
            At.colIndex[At.rowPtr[A.colIndex[j] + 1]] = i;
            At.rowPtr[A.colIndex[j] + 1]++;
        }
}

void TransposeMatrix2(CRSMatrix& A, CRSMatrix& At)
{
    At.val.clear();
    At.colIndex.clear();
    At.rowPtr.clear();
    At.n = A.m;
    At.m = A.n;
    At.nz = A.nz;
    std::vector<std::vector<double>> values;
    std::vector<std::vector<int>> index;
    values.resize(A.m);
    index.resize(A.m);

    for(int i = 0; i < A.n; i++)
    {
        for(int k = A.rowPtr[i]; k < A.rowPtr[i + 1]; k++)
        {
            values[A.colIndex[k]].push_back(A.val[k]);
            index[A.colIndex[k]].push_back(i);
        }
    }
    int sum = 0;
    for(int i = 0; i < A.m; i++)
    {
        for(int j = 0; j < values[i].size(); j++)
        {
            At.val.push_back(values[i][j]);
            At.colIndex.push_back(index[i][j]);
        }
        At.rowPtr.push_back(sum);
        sum += values[i].size();
    }
    At.rowPtr.push_back(A.nz);
}

void SLE_Solver_CRS_BICG(CRSMatrix& A, double* b, double eps, int max_iter, double* x, int& count)
{
    CRSMatrix At;
    TransposeMatrix(A, At);

    double* r = new double[A.n];
    double* bi_r = new double[A.n];
    double* n_r = new double[A.n];
    double* nbi_r = new double[A.n];

    double* p = new double[A.n];
    double* bi_p = new double[A.n];
    double* n_p = new double[A.n];
    double* nbi_p = new double[A.n];

    double* A_p = new double[A.n];
    double* Atbi_p = new double[A.n];

    memset(x, 1., sizeof(double)*A.n);

    MultMatrixVector(A, x, A_p);
#pragma omp parallel for
    for(int i = 0; i < A.n; i++)
        r[i] = bi_r[i] = p[i] = bi_p[i] = b[i] - A_p[i];
    for(count = 0; count < max_iter; count++)
    {
        MultMatrixVector(A, p, A_p);
        MultMatrixVector(At, bi_p, Atbi_p);
        double numerator = MultVectorVector(A.n, bi_r, r);
        double denominator = MultVectorVector(A.n, bi_p, A_p);
        double alpha = numerator / denominator;

        double check = 0.;
#pragma omp parallel for reduction(+:check)
        for(int i = 0; i < A.n; ++i)
        {
            double add = alpha * p[i];
            x[i] += add;
            check += add * add;
        }
        if(check < eps*eps)
            break;
#pragma omp parallel for
        for(int i = 0; i < A.n; i++)
        {
            n_r[i] = r[i] - alpha * A_p[i];
            nbi_r[i] = bi_r[i] - alpha * Atbi_p[i];
        }
        denominator = numerator;
        numerator = MultVectorVector(A.n, nbi_r, n_r);
        double beta = numerator / denominator;
        if(beta == 0. || MultVectorVector(A.n, nbi_r, nbi_r) < eps*eps)
            break;
#pragma omp parallel for
        for(int i = 0; i < A.n; i++)
        {
            n_p[i] = n_r[i] + beta * p[i];
            nbi_p[i] = nbi_r[i] + beta * bi_p[i];
        }
        std::swap(r, n_r);
        std::swap(p, n_p);
        std::swap(bi_r, nbi_r);
        std::swap(bi_p, nbi_p);
    }
}

void InitializeMatrix(int N, int NZ, CRSMatrix &mtx)
{
    mtx.n = N;
    mtx.nz = NZ;
    mtx.val.resize(NZ);
    mtx.colIndex.resize(NZ);
    mtx.rowPtr.resize(N + 1);
}

double next()
{
    return ((double)rand() / (double)RAND_MAX);
}

void GenerateRegularCRS(int seed, int N, int cntInRow, CRSMatrix& mtx)
{
    int i, j, k, f, tmp, notNull, c;

    srand(seed);

    notNull = cntInRow * N;
    InitializeMatrix(N, notNull, mtx);

    for(i = 0; i < N; i++)
    {
        // Формируем номера столбцов в строке i
        for(j = 0; j < cntInRow; j++)
        {
            do
            {
                mtx.colIndex[i * cntInRow + j] = rand() % N;
                f = 0;
                for(k = 0; k < j; k++)
                    if(mtx.colIndex[i * cntInRow + j] == mtx.colIndex[i * cntInRow + k])
                        f = 1;
            } while(f == 1);
        }
        // Сортируем номера столцов в строке i
        for(j = 0; j < cntInRow - 1; j++)
            for(k = 0; k < cntInRow - 1; k++)
                if(mtx.colIndex[i * cntInRow + k] > mtx.colIndex[i * cntInRow + k + 1])
                {
                    tmp = mtx.colIndex[i * cntInRow + k];
                    mtx.colIndex[i * cntInRow + k] = mtx.colIndex[i * cntInRow + k + 1];
                    mtx.colIndex[i * cntInRow + k + 1] = tmp;
                }
    }

    // Заполняем массив значений
    for(i = 0; i < cntInRow * N; i++)
        mtx.val[i] = next() * 100.;

    // Заполняем массив индексов строк
    c = 0;
    for(i = 0; i <= N; i++)
    {
        mtx.rowPtr[i] = c;
        c += cntInRow;
    }
}

void InitializeVector(int N, double **vec)
{
    *vec = new double[N];
}

void GenerateVector(int seed, int N, double **vec)
{
    srand(seed);
    InitializeVector(N, vec);
    for(int i = 0; i < N; i++)
    {
        (*vec)[i] = next() * 100.;
    }
}

int main()
{
    setlocale(LC_ALL, "rus");
    std::ofstream fout;
    fout.open("C:\\Users\\kochu\\Desktop\\ResultPP2.csv", std::ios_base::out);
    std::cout << "MAX потоков = " << omp_get_max_threads() << "\n";
    fout << "Количесво потоков;" << "Размер матрицы;" << "Время работы;" << "Количество итераций;" << "\n";
    int n = 1000;
    while(n <= 10000)
    {
        std::cout << "Genereting matrix..." << std::endl;
        double* b;
        CRSMatrix A;
        int count;
        GenerateRegularCRS(20, n, n*0.1, A);
        GenerateVector(20, n, &b);
        double* x = new double[A.n];
        for(int i = 0; pow(2, i) < pow(2, 3); i++)
        {
            std::cout << "Количесво потоков = " << pow(2, i) << std::endl;
            omp_set_num_threads(pow(2, i));
            fout << pow(2, i) << ";";
            fout << n << ";";
            std::cout << "SLE_Solver_CRS_BICG..." << std::endl;
            double tStart = omp_get_wtime();
            SLE_Solver_CRS_BICG(A, b, 0.000001, 100, x, count);
            double tFinish = omp_get_wtime();
            std::cout << count << std::endl;
            std::cout << tFinish - tStart << std::endl;
            fout << tFinish - tStart << ";";
            fout << count << ";";
            fout << "\n";
        }
        n += 1000;
    }
    fout.close();
    std::cout << "Данные успешно записаны в файл ResultPP2.csv\n";
    system("pause");
    return 0;
}
