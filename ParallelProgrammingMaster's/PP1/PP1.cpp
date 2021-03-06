// PP1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <omp.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

double* reductMatrix(double* A, double* B, int n)
{
    double* redM = new double[n*n];
#pragma omp parallel for
    for (int i = 0; i < n*n; ++i)
        redM[i] = A[i] - B[i];
    return redM;
}

double* GetPart(double* A, int i, int i2, int j, int j2, int size)
{
    double* part = new double[(i2 - i)*(j2 - j)];
    for (int k = i; k < i2; ++k)
        for (int l = j; l < j2; ++l)
            part[k - i + (l - j)*(i2 - i)] = A[k + l * size];
    return part;
}

void resolveTriangleSystem(double* A, double* x, double* b, int n)
{
    x[0] = b[0] / A[0];
    for (int i = 1; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)
            b[i] -= A[j + i * n] * x[j];
        x[i] = b[i] / A[i + i * n];
    }
}

void resolveMatrSystem(double* A, double* X, double* B, int n, int m)
{
#pragma omp parallel for
    for (int i = 0; i < m; ++i)
        resolveTriangleSystem(A, X + i * n, B + i * n, n);
}

void L11_L21toL(double* L, double* L11, double* L21, int n, int r)
{
    memset(L, 0, sizeof(double)*n*n);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < r; ++j)
            L[j + i * n] = L11[j + i * r];
    for (int i = r; i < n; ++i)
        for (int j = 0; j < r; ++j)
            L[j + i * n] = L21[j + (i - r)*r];

}
void L22toL(double* L, double* L22, int n, int r)
{
    for (int i = r; i < n; ++i)
        for (int j = r; j < n; ++j)
            L[j + i * n] = L22[(j - r) + (i - r)*(n - r)];
}
void Origin_Cholesky_Decomposition(double* A, double* L, int n)
{
    memset(L, 0, sizeof(double)*n*n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j)
            {
                double sum = 0;
                for (int k = 0; k < i; ++k)
                    sum += L[k + i * n] * L[k + i * n];
                L[i + i * n] = std::sqrt(A[i + i * n] - sum);
            }
            else if (j < i)
            {
                double sum = 0;
                for (int k = 0; k < j; ++k)
                    sum += L[k + i * n] * L[k + j * n];
                L[j + i * n] = (A[j + i * n] - sum) / L[j + j * n];
            }
        }
    }
}

void Origin_Cholesky_Decomposition_2(double* A, double* L, int n)
{
    memset(L, 0, sizeof(double)*n*n);
    for(int i = 0; i < n; i++)
    {
        L[i + i * n] = A[i + i * n];
        for(int k = 0; k < i; k++)
            L[i + i * n] -= L[k + i * n] * L[k + i * n];
        L[i + i * n] = std::sqrt(L[i + i * n]);
        for(int j = i + 1; j < n; j++)
        {
            L[i + j * n] = A[i + j * n];
            for(int k = 0; k < i; k++)
                L[i + j * n] -= L[k + i * n] * L[k + j * n];
            L[i + j * n] /= L[i + i * n];
        }
    }
}

void MultMatrix(double *A, double *B, double *C, const int n, int m)
{
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < m; ++k)
                C[i*n + j] += A[i*m + k] * B[j*m + k];
}

void Cholesky_Decomposition(double* A, double* L, int n)
{
    int r;
    if (n > 500)
        r = 500;
    else
    {
        Origin_Cholesky_Decomposition(A, L, n);
        return;
    }

    double* L11 = new double[r*r];
    double* A11 = GetPart(A, 0, r, 0, r, n);
    Origin_Cholesky_Decomposition(A11, L11, r);
    delete[] A11;

    double* A21 = GetPart(A, 0, r, r, n, n);
    double* L21 = new double[r * (n - r)];
    resolveMatrSystem(L11, L21, A21, r, n - r);
    delete[] A21;

    double* A22 = GetPart(A, r, n, r, n, n);
    double* L22 = new double[(n - r) * (n - r)];
    double* resMult = new double[(n - r) * (n - r)];
    MultMatrix(L21, L21, resMult, n - r, r);
    double* redA22 = reductMatrix(A22, resMult, n - r);
    delete[] resMult;
    delete[] A22;

    L11_L21toL(L, L11, L21, n, r);
    delete[] L11;
    delete[] L21;
    if (n - r > r)
        Cholesky_Decomposition(redA22, L22, n - r);
    else
        Origin_Cholesky_Decomposition(redA22, L22, n - r);
    L22toL(L, L22, n, r);
    delete[] L22;
    delete[] redA22;
}
int main()
{
    setlocale(LC_ALL, "rus");
    std::ofstream fout;
    fout.open("C:\\Users\\kochu\\Desktop\\Result.csv", std::ios_base::out);
    std::cout << "MAX потоков = " << omp_get_max_threads() << "\n";
    for(int i = 0; pow(2, i) < pow(2, 3); i++)
    {
        std::cout << "Количесво потоков = " << pow(2, i) << std::endl;
        omp_set_num_threads(pow(2, i));
        fout << "Количесво потоков = " << omp_get_num_threads() << "\n";
        fout << "Размер матрицы;" << "Холецкий 1;" << "Холецкий 2;" << "Холецкий блочный;" << "\n";

        int n = 1000;
        while(n <= 5000)
        {
            std::cout << "Размер матрицы = " << n << std::endl;
            double* A = new double[n * n];
            double* L;
            L = new double[n * n];

            A[0] = 0;
            for(int i = 0; i < n; i++)
            {
                for(int j = i; j < n; j++)
                {
                    if(j != 0)
                    {
                        A[i * n + j] = A[i * n + j - 1] + 1;
                        A[j * n + i] = A[i * n + j - 1] + 1;
                    }
                }
            }

            double sum = 0;
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    sum += A[i * n + j];
                }
                A[i * n + i] = sum / 2;
                sum = 0;
            }
            fout << n << ";";

            double tStart, tFinish, tChol1, tChol2, tBlock;
            std::cout << "Холецкий 1\n";
            if(pow(2, i) == 1)
            {
                tStart = omp_get_wtime();
                Origin_Cholesky_Decomposition(A, L, n);
                tFinish = omp_get_wtime();
                tChol1 = (tFinish - tStart);
            }
            else
                fout << 0.;
            fout << tChol1 << ";";

            std::cout << "Холецкий 2\n";
            if(pow(2, i) == 1)
            {
                tStart = omp_get_wtime();
                Origin_Cholesky_Decomposition_2(A, L, n);
                tFinish = omp_get_wtime();
                tChol2 = (tFinish - tStart);
            }
            else
                fout << 0.;
            fout << tChol2 << ";";

            std::cout << "Холецкий блочный\n";
            tStart = omp_get_wtime();
            Cholesky_Decomposition(A, L, n);
            tFinish = omp_get_wtime();
            tBlock = (tFinish - tStart);

            fout << tBlock << ";";

            fout << "\n";
            n += 1000;
            delete[] A;
            delete[] L;
        }

    }
    fout.close();
    std::cout << "Данные успешно записаны в файл Result.csv\n";   
    system("pause");
    return 0;
};

