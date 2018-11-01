#include "stdafx.h"
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <fstream>
#include <omp.h>

//sin
class heat_task
{
public:
    double T;
    double L;
    int n;
    int m;
    heat_task(double L, double T)
    {
        this->L = L;
        this->T = T;
    }

    virtual double initial_condition(double x)
    {
        return u(x, 0);
    }

    virtual double left_condition(double t)
    {
        return u(0, t);
    }

    virtual double right_condition(double t)
    {
        return u(L, t);
    }

    virtual double f(double x, double t)
    {
        return x * std::cos(x * t) + t * t * std::sin(x * t);
    }

    virtual double u(double x, double t)
    {
        return std::sin(x * t);
    }
};

//test6
/*class heat_task
{
public:
    double T;
    double L;
    int n;
    int m;
    heat_task(double L, double T)
    {
        this->L = L;
        this->T = T;
    }

    virtual double initial_condition(double x)
    {
        return u(x, 0);
    }

    virtual double left_condition(double t)
    {
        return u(0, t);
    }

    virtual double right_condition(double t)
    {
        return u(L, t);
    }

    virtual double f(double x, double t)
    {
        return 0.5 * ((_j0(t) - _jn(2, t)) * _j0(x) - _j1(t) * (_jn(2, x) - _j0(x)));
    }

    virtual double u(double x, double t)
    {
        return _j1(t) * _j0(x);
    }
};*/

//test7
/*class heat_task
{
public:
    double T;
    double L;
    int n;
    int m;
    heat_task(double L, double T)
    {
        this->L = L;
        this->T = T;
    }

    virtual double initial_condition(double x)
    {
        return u(x, 0);
    }

    virtual double left_condition(double t)
    {
        return u(0, t);
    }

    virtual double right_condition(double t)
    {
        return u(L, t);
    }

    virtual double f(double x, double t)
    {
        return (_j0(t) - _jn(2, t)) * exp(_j1(t) / _j0(x)) / (2 * _j0(x)) - _j1(t) * exp(_j1(t) / _j0(x)) * (2 * _j1(t) * _j1(x) * _j1(x) + _j0(x) * _j0(x) * _j0(x) - _jn(2, x) * _j0(x) * _j0(x) + 4 * _j1(x) * _j1(x) * _j0(x)) / (2 * pow(_j0(x), 4));
    }

    virtual double u(double x, double t)
    {
        return exp(_j1(t) / _j0(x));
    }
};*/

class CThreeDiagonalMatrix
{

    int m_matrixSize = 0;
    int m_chunckSize = 0;
    int m_numberOfThreads = 0;
    double* m_diagonal = nullptr;
    double* m_overDiagonal = nullptr;
    double* m_subDiagonal = nullptr;
    double* m_alphaCoeffs = nullptr;
    double* m_betaCoeffs = nullptr;
    
public:
    explicit CThreeDiagonalMatrix(int matrixSize);
    void updateMatrix(const double coeffA, const double coeffB);
    void algorithmParallel(double* function, double* result);
};

CThreeDiagonalMatrix::CThreeDiagonalMatrix(int matrixSize) : m_matrixSize(matrixSize)
{
    m_numberOfThreads = omp_get_num_threads();
    if(m_numberOfThreads > 1)
    {
        m_alphaCoeffs = new double[m_numberOfThreads - 1];
        m_betaCoeffs = new double[m_numberOfThreads - 1];
    }
    m_chunckSize = m_matrixSize / m_numberOfThreads;
    m_diagonal = new double[m_matrixSize];
    m_overDiagonal = new double[m_matrixSize - 1];
    m_subDiagonal = new double[m_matrixSize - 1];
}

void CThreeDiagonalMatrix::updateMatrix(const double coeffA, const double coeffB)
{
#pragma omp parallel for
    for(int i = 0; i < m_matrixSize; ++i)
    {

        m_diagonal[i] = coeffA;
        if(i < m_matrixSize - 1)
        {
            m_overDiagonal[i] = coeffB;
            m_subDiagonal[i] = coeffB;
        }
    }
}

void CThreeDiagonalMatrix::algorithmParallel(double* function, double* result)
{
#pragma omp parallel
    {
        const int threadNum = omp_get_thread_num();
        if(threadNum == 0)
        {
            for(int i = 1; i < m_chunckSize; ++i)
            {
                m_diagonal[i] -= m_subDiagonal[i - 1] *m_overDiagonal[i - 1] / m_diagonal[i - 1];
                function[i] -= m_subDiagonal[i - 1] *function[i - 1] / m_diagonal[i - 1];
                m_subDiagonal[i - 1] = 0.;
            }
        }
        if((threadNum > 0) && (threadNum < m_numberOfThreads - 1)) 
        {
            for(int i = m_chunckSize*threadNum + 1; i < m_chunckSize*(threadNum + 1); ++i)
            {
                m_diagonal[i] -= m_subDiagonal[i - 1] *m_overDiagonal[i - 1] / m_diagonal[i - 1];
                function[i] -= m_subDiagonal[i - 1] *function[i - 1] / m_diagonal[i - 1];
                m_subDiagonal[i - 1] = -1.0*m_subDiagonal[i - 1] *m_subDiagonal[i - 2] / m_diagonal[i - 1];
            }
        }
        if((threadNum == m_numberOfThreads - 1) && (m_numberOfThreads > 1)) 
        {
            for(int i = m_chunckSize*(m_numberOfThreads - 1) + 1; i < m_matrixSize; ++i)
            {
                m_diagonal[i] -= m_subDiagonal[i - 1] *m_overDiagonal[i - 1] / m_diagonal[i - 1];
                function[i] -= m_subDiagonal[i - 1] *function[i - 1] / m_diagonal[i - 1];
                m_subDiagonal[i - 1] = -1.0*m_subDiagonal[i - 1] *m_subDiagonal[i - 2] / m_diagonal[i - 1];
            }
        }
#pragma omp barrier
        if(threadNum == 0)
        {
            for(int i = m_chunckSize - 3; i >= 0; --i)
            {
                function[i] -= m_overDiagonal[i] * function[i + 1] / m_diagonal[i + 1];
                m_overDiagonal[i] = -m_overDiagonal[i] *m_overDiagonal[i + 1] / m_diagonal[i + 1];
                
            }
        }
        if((threadNum > 0) && (threadNum < m_numberOfThreads - 1))
        {
            const int chuckEnd = m_chunckSize*threadNum - 1;
            for(int i = m_chunckSize * (threadNum + 1) - 3; i >= chuckEnd; --i)
            {
                function[i] -= m_overDiagonal[i] * function[i + 1] / m_diagonal[i + 1];
                if(i != chuckEnd)
                    m_subDiagonal[i - 1] -= m_overDiagonal[i] *m_subDiagonal[i] / m_diagonal[i + 1];
                if(i == chuckEnd)
                    m_diagonal[i] -= m_overDiagonal[i] *m_subDiagonal[i] / m_diagonal[i + 1];
                m_overDiagonal[i] = -m_overDiagonal[i] * m_overDiagonal[i + 1] / m_diagonal[i + 1];
            }
        }
        if((threadNum == m_numberOfThreads - 1) && (m_numberOfThreads > 1))
        {
            const int chuckEnd = m_chunckSize*threadNum - 1;
            for(int i = m_matrixSize - 3; i >= chuckEnd; --i)
            {
                function[i] -= m_overDiagonal[i] * function[i + 1] / m_diagonal[i + 1];
                if(i != chuckEnd)
                    m_subDiagonal[i - 1] -= m_overDiagonal[i] *m_subDiagonal[i] / m_diagonal[i + 1];
                if(i == chuckEnd)
                    m_diagonal[i] -= m_overDiagonal[i] *m_subDiagonal[i] / m_diagonal[i + 1];
                m_overDiagonal[i] = -m_overDiagonal[i] * m_overDiagonal[i + 1] / m_diagonal[i + 1];
            }
        }
#pragma omp barrier
        if(threadNum == 0)
        {
            if(m_numberOfThreads == 1)
            {
                result[m_matrixSize] = function[m_matrixSize - 1] / m_diagonal[m_matrixSize - 1];
            }
            else
            {
                const int coeffsSize = m_numberOfThreads - 1;
                m_alphaCoeffs[0] = -1.0*m_overDiagonal[m_chunckSize - 1] / m_diagonal[m_chunckSize - 1];
                m_betaCoeffs[0] = function[m_chunckSize - 1] / m_diagonal[m_chunckSize - 1];
                for(int i = 1; i < coeffsSize; ++i)
                {
                    const int pos = (i + 1)*m_chunckSize - 1;
                    m_alphaCoeffs[i] = -1.0*m_overDiagonal[pos] / (m_alphaCoeffs[i - 1] * m_subDiagonal[pos - 1] + m_diagonal[pos]);
                    m_betaCoeffs[i] = (function[pos] - m_betaCoeffs[i - 1] * m_subDiagonal[pos - 1]) / (m_alphaCoeffs[i - 1] * m_subDiagonal[pos - 1] + m_diagonal[pos]);
                }
                result[m_matrixSize] = (function[m_matrixSize - 1] - m_betaCoeffs[coeffsSize - 1] * m_subDiagonal[m_matrixSize - 2]) / (m_alphaCoeffs[coeffsSize - 1] * m_subDiagonal[m_matrixSize - 2] + m_diagonal[m_matrixSize - 1]);
                result[coeffsSize*m_chunckSize] = m_alphaCoeffs[coeffsSize - 1] * result[m_matrixSize] + m_betaCoeffs[coeffsSize - 1];
                for(int i = coeffsSize - 1; i > 0; --i)
                    result[i*m_chunckSize] = m_alphaCoeffs[i - 1] * result[(i + 1)*m_chunckSize] + m_betaCoeffs[i - 1];
            }
        }

#pragma omp barrier
        if(threadNum == 0)
            for(int i = 0; i < m_chunckSize - 1; ++i)
                result[i + 1] = (function[i] - m_overDiagonal[i]*result[m_chunckSize]) / m_diagonal[i];
        if((threadNum > 0) && (threadNum < m_numberOfThreads - 1))
        {
            const int chuckStart = m_chunckSize*threadNum;
            const int chuckEnd = m_chunckSize * (threadNum + 1);
            for(int i = chuckStart; i < chuckEnd - 1; ++i)
                result[i + 1] = (function[i] - m_subDiagonal[i - 1]*result[chuckStart] - m_overDiagonal[i]*result[chuckEnd]) / m_diagonal[i];
        }
        if((threadNum == m_numberOfThreads - 1) && (m_numberOfThreads > 1))
        {
            const double vn = result[m_matrixSize];
            const int chuckStart = m_chunckSize * threadNum;
            for(int i = chuckStart; i < m_matrixSize - 1; ++i)
                result[i + 1] = (function[i] - m_subDiagonal[i - 1]*result[chuckStart] - m_overDiagonal[i]*vn) / m_diagonal[i];
        }
    }
}

void updateF(double* f, const double* v, heat_task& task, const double coeffB, const double coeffOther, const double tau, const int j, const double h, const int matrixSize, const double tauj)
{
    const double tauhh = 0.5*(tau / (h*h));
#pragma omp parallel for
    for(int i = 1; i <= matrixSize; ++i)
        f[i - 1] = coeffOther*v[i] + tauhh*(v[i - 1] + v[i + 1]) + tau*task.f(h*i, (j - 0.5)*tau);
    f[0] -= coeffB*task.left_condition(tauj);
    f[matrixSize - 1] -= coeffB*task.right_condition(tauj);
}
void heat_equation_crank_nicolson(heat_task task, double* v)
{
    if(task.n == 1)
    {
        v[0] = task.left_condition(task.T);
        v[1] = task.right_condition(task.T);
        return;
    }
    const double h = task.L / task.n;
    const double tau = task.T / task.m;
    double* f = new double[task.n - 1];
    for(int i = 0; i < task.n + 1; ++i)
        v[i] = task.initial_condition(h*i);
    CThreeDiagonalMatrix matrix(task.n - 1);
    const double coeffA = 1.0 + (tau / (h*h));
    const double coeffB = -0.5 * (tau / (h*h));
    const double coeffOther = 1.0 - (tau / (h*h));
    const int matrixSize = task.n - 1;
    double tauj;
    for(int j = 1; j <= task.m; ++j)
    {
        tauj = tau*j;
        matrix.updateMatrix(coeffA, coeffB);
        updateF(f, v, task, coeffB, coeffOther, tau, j, h, matrixSize, tauj);
        v[0] = task.left_condition(tauj);
        v[task.n] = task.right_condition(tauj);
        if(matrixSize == 1)
            v[matrixSize] = f[0] / coeffA;
        if(matrixSize > 1)
            matrix.algorithmParallel(f, v);
    }
    delete[] f;
}

//for test
double calcErr(heat_task task, double* v)
{
    const double h = task.L / task.n;
    double deltaNumeric = 0.;
    for(int i = 0; i < task.n + 1; ++i)
    {
        double sub = std::abs(task.u(i * h, task.T) - v[i]);
        if(sub > deltaNumeric)
            deltaNumeric = sub;
    }
    return deltaNumeric;
}

double calcConvErr(heat_task& task1, double* v1, heat_task& task2, double* v2)
{
    const double h_1 = task1.L / task1.n;
    const double tau_1 = task1.T / task1.m;
    const double h_2 = task2.L / task2.n;
    const double tau_2 = task2.T / task2.m;
    const double deltaTheory1 = h_1 * h_1 + tau_1 * tau_1;
    const double deltaTheory2 = h_2 * h_2 + tau_2 * tau_2;
    const double RTheory = deltaTheory1 / deltaTheory2;
    const double deltaNumeric1 = calcErr(task1, v1);
    const double deltaNumeric2 = calcErr(task2, v2);
    const double RNumeric = deltaNumeric1 / deltaNumeric2;
    return std::abs(RTheory - RNumeric);
}

void RunTest(double L, double T, std::ofstream& fout)
{
    const double epsConv = 0.1;
    const double eps = 0.000001;
    double* v = nullptr;
    double* v2 = nullptr;
    for(int i = 1; i < 12; ++i)
    {
        if(i == 1)
        {
            const int n1 = pow(2, 2 + i);
            const int m1 = pow(2, 2 + i);
            const int n2 = pow(2, 3 + i);
            const int m2 = pow(2, 3 + i);
            v = new double[n1 + 1];
            v2 = new double[n2 + 1];
            memset(v, 0., sizeof(double) * (n1 + 1));
            memset(v2, 0., sizeof(double) * (n2 + 1));
            heat_task task1(L, T);
            task1.n = n1;
            task1.m = m1;
            fout << n1 << ";";
            double tStart = omp_get_wtime();
            heat_equation_crank_nicolson(task1, v);
            double tEnd = omp_get_wtime();
            fout << tEnd - tStart << ";" << "\n";
            heat_task task2(L, T);
            task2.n = n2;
            task2.m = m2;
            fout << ";" << n2 << ";";
            tStart = omp_get_wtime();
            heat_equation_crank_nicolson(task2, v2);
            tEnd = omp_get_wtime();
            fout << tEnd - tStart << ";" << "\n";
            const double errConv = calcConvErr(task1, v, task2, v2);
            std::cout << "conv (" << n1 << ", " << m1 << ")(" << n2 << ", " << m2 << ((errConv < epsConv) ? ") correct" : ") not correct") << ", err: " << errConv << std::endl;
            const double err1 = calcErr(task1, v);
            std::cout << "err (" << n1 << ", " << m1 << ") " << (err1 < eps ? " correct" : " not correct") << ", err: " << err1 << std::endl;
            const double err2 = calcErr(task2, v2);
            std::cout << "err (" << n2 << ", " << m2 << ") " << (err2 < eps ? " correct" : " not correct") << ", err: " << err2 << std::endl;
            delete[] v;
        }
        else
        {
            const int n = pow(2, 3 + i);
            const int m = pow(2, 3 + i);
            const int n2 = pow(2, 2 + i);
            const int m2 = pow(2, 2 + i);
            v = new double[n + 1];
            memset(v, 0., sizeof(double) * (n + 1));
            heat_task task(L, T);
            task.n = n;
            task.m = m;
            fout << ";" << n << ";";
            double tStart = omp_get_wtime();
            heat_equation_crank_nicolson(task, v);
            double tEnd = omp_get_wtime();
            fout << tEnd - tStart << ";" << "\n";
            heat_task task2(L, T);
            task2.n = n2;
            task2.m = m2;
            double errConv = calcConvErr(task2, v2, task, v);
            std::cout << "conv (" << n2 << ", " << m2 << ")(" << n << ", " << m << (errConv < epsConv ? ") correct" : ") not correct") << ", err: " << errConv << std::endl;
            double err = calcErr(task, v);
            std::cout << "err (" << n << ", " << m << ") " << (err < eps ? " correct" : " not correct") << ", err: " << err << std::endl;
            delete[] v2;
            v2 = v;
        }
    }
    delete[] v2;
}
int main()
{
    setlocale(LC_ALL, "rus");
    std::ofstream fout;
    fout.open("C:\\Users\\kochu\\Desktop\\ResultPP3.csv", std::ios_base::out);
    std::cout << "MAX потоков = " << omp_get_max_threads() << "\n";
    fout << "Количесво потоков;" << "Размер сетки;" << "Время работы;" << "\n";
    for(int i = 0; pow(2, i) < pow(2, 3); i++)
    {
        std::cout << "Количесво потоков = " << pow(2, i) << std::endl;
        omp_set_num_threads(pow(2, i));
        fout << pow(2, i) << ";";
        double L = 1.0;
        double T = 1.0;
        RunTest(L, T, fout);
    }
    fout.close();
    std::cout << "Данные успешно записаны в файл ResultPP3.csv\n";
    system("pause");
    return 0;
}