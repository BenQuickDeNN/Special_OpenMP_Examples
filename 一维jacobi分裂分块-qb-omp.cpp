/**
 * 分裂分块对照论文《Split Tiling for GPUs Automatic Parallelization Using Trapezoidal Tiles》图3实现
 * 编译时注意使用多线程和向量化，否则可能会因为算例不够而导致局部性优化效果不明显
 */ 

typedef float type;

/* 定义数据，ghost cell 大小为1 */
const int I = 8 * 80000; // 数组大小
const int Ig = I + 2; // 实际数组大小
const int TSTEP = 10000; // 迭代次数，要选择偶数
const type ERROR = 0.1; // 误差
type A[2][Ig], B[2][Ig];

/* 设置线程数 */
const int NT = 4;

/* 设置分块大小 */
const int Sup = 1000; // 分块梯形上底长度
const int Sdown = 4000; // 分块梯形下底长度
const int LEN = Sup + Sdown; // 上底加下底的长度

/**
 * @brief 初始化ghost cell
 */
inline void init_ghost_cell();

/**
 * @brief 使用随机数初始化数组
 */
inline void init_array();

/**
 * @brief 执行baseline组
 */
inline void exec_baseline();

/**
 * @brief 执行分裂分块组
 */
inline void exec_split_tiling();

/**
 * @brief 验证计算结果正确性
 */
void validate_array();

#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <time.h>

using namespace std;

int main()
{
    /* 初始化数据 */
    init_ghost_cell();
    init_array();

    /* 初始化计时器 */
    clock_t timmer;

    /* 执行baseline组 */
    timmer = clock();
    exec_baseline();
    timmer = clock() - timmer;
    fprintf(stdout, "baseline elapsed %f s\n", (float)(timmer) / (float)CLK_TCK);

    /* 执行分裂分块组 */
    timmer = clock();
    exec_split_tiling();
    timmer = clock() - timmer;
    fprintf(stdout, "split tiling elapsed %f s\n", (float)(timmer) / (float)CLK_TCK);

    /* 验证结果 */
    validate_array();

    return 0;
}

inline void init_ghost_cell()
{
    A[0][0] = 0; A[0][I + 1] = 0;
    A[1][0] = 0; A[1][I + 1] = 0;
    B[0][0] = 0; B[0][I + 1] = 0;
    B[1][0] = 0; B[1][I + 1] = 0;
}

inline void init_array()
{
    const type scale = 2.0;
    #pragma omp parallel for
    for (int i = 1; i <= I; i++)
    {
        A[0][i] = ((type) rand()) / scale;
        B[0][i] = A[0][i];
    }
}

inline void exec_baseline()
{
    for (int t = 0; t < TSTEP; t++)
    {
        #pragma omp parallel for num_threads(NT)
        for (int i = 1; i <= I; i++)
            A[(t + 1) % 2][i] = 0.5 * (A[t % 2][i - 1] + A[t % 2][i + 1]);
    }
}

inline void exec_split_tiling()
{
    /* 计算时间维分块长度 */
    const int St = (Sdown - Sup) / 2 + 1;
    
    for (int tt = 0; tt < TSTEP; tt += St)
    {
        /* 先计算绿色分块 */
        #pragma omp parallel for num_threads(NT)
        for (int ii = 1; ii <= I; ii += LEN)
            for (int t = tt; t < min(tt + St, TSTEP); t++)
            {
                const int offset1 = t - tt;
                const int offset2 = Sdown - offset1 - 1;
                const int iStart = ii + offset1;
                const int iEnd =  min(ii + offset2, I);
                for (int i = iStart; i <= iEnd; i++)
                    B[(t + 1) % 2][i] = 0.5 * (B[t % 2][i - 1] + B[t % 2][i + 1]);
            }

        /* 再计算蓝色分块 */
        const int II = I + LEN;
        #pragma omp parallel for num_threads(NT)
        for (int ii = 1 + Sdown; ii <= II; ii += LEN)
            for (int t = tt; t < min(tt + St, TSTEP); t++)
            {
                const int offset1 = t - tt;
                const int offset2 = Sup + offset1 - 1;
                const int iStart = ii - offset1;
                const int iEnd = min(ii + offset2, I);
                for (int i = iStart; i <= iEnd; i++)
                    B[(t + 1) % 2][i] = 0.5 * (B[t % 2][i - 1] + B[t % 2][i + 1]);
            }

        /* 计算蓝色分块边缘碎片 */
        for (int t = tt + 1; t < min(tt + St, TSTEP); t++)
            for (int i = 1; i <= t - tt; i++)
                B[(t + 1) % 2][i] = 0.5 * (B[t % 2][i - 1] + B[t % 2][i + 1]);
    }
}

void validate_array()
{
    for (int i = 1; i <= I; i++)
        if (fabs(A[0][i] - B[0][i]) > ERROR)
        {
            fprintf(stderr, "computation incorrected!\n");
            return;
        }
    fprintf(stderr, "computation corrected!\n");
}