#include <omp.h>

const int X = 1000;
const int Y = 1000;
const int XS = X + 1;
const int YS = Y + 1;

/* 定义数据 */
float* DATA1;
float* DATA2;

const int STEPs = 100;
const int NUM_THREADS = 8;
const float error = 0.001;
const float def_val = 0.01; // 值、数据范围、迭代步不能过大，否则可能因为溢出而导致结果错误或程序崩溃

/* 串行版本 */
void kernel_serial(float *data)
{
    for (int t = 0; t < STEPs; t++)
        for (int x = 1; x < XS; x++)
            for (int y = 1; y < YS; y++)
                data[x * YS + y] = data[(x - 1) * YS + y] + data[x * YS + y - 1];
}

/* 并行版本 */
/* 使用倾斜、波阵面并行处理，并行执行方向与Y轴平行 */
void kernel_skew(float *data)
{
    const int blockX = XS / NUM_THREADS; // 计算每个线程处理的X维数据范围
    omp_lock_t lock[NUM_THREADS - 1]; // 定义互斥锁
    int execPoints[NUM_THREADS - 1]; // 定义执行点数
    for (int i = 0; i < NUM_THREADS - 1; i++)
    {
        omp_init_lock(&lock[i]); // 初始化互斥锁
        execPoints[i] = 0; // 初始化执行点
    }
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        const int tId = omp_get_thread_num(); // 获取当前线程编号
        const int Xstart = tId * blockX + 1;
        int Xend = Xstart + blockX;
        if (tId == NUM_THREADS - 1) // 最后一个线程要处理不对齐的那部分
            Xend = XS;
        
        /* 经过循环倾斜（延迟）处理后的内核 */
        for (int t = 0; t < STEPs; t++)
            for (int x = Xstart; x < Xend; x++)
                for (int y = 1; y < YS; y++)
                {
                    // 消耗上一个线程提供的执行点
                    if (tId > 0 && x == Xstart)
                    {
                        omp_set_lock(&lock[tId - 1]); // 获取互斥锁
                        if (execPoints[tId - 1] <= 0)
                        { 
                            omp_unset_lock(&lock[tId - 1]); //释放互斥锁
                            /* 等待上一个线程增加执行点 */ 
                            y--;
                            continue;
                        }
                        else
                        {
                            execPoints[tId - 1]--; // 消耗执行点
                            omp_unset_lock(&lock[tId - 1]); //释放互斥锁
                        }
                    }
                    data[x * YS + y] = data[(x - 1) * YS + y] + data[x * YS + y - 1];
                    if (tId < NUM_THREADS - 1 && x == Xend - 1)
                    {
                        omp_set_lock(&lock[tId]); // 获取互斥锁
                        execPoints[tId]++; // 增加执行点
                        omp_unset_lock(&lock[tId]); //释放互斥锁
                    }
                }
    }
    for (int i = 0; i < NUM_THREADS - 1; i++)
        omp_destroy_lock(&lock[i]); // 销毁互斥锁
}

/* 初始化数据 */
void initialize(float *data, const float& val)
{
    #pragma omp parallel for
    for (int x = 0; x < XS; x++)
        for (int y = 0; y < YS; y++)
            data[x * YS + y] = val;
}

/* 验证结果 */
#include <cmath>
bool verify(float *data1, float *data2)
{
    for (int x = 0; x <= X; x++)
        for (int y = 0; y <= Y; y++)
            if (std::abs(data1[x * YS + y] - data2[x * YS + y]) > error)
                return false;
    return true;
}

#include <time.h>
#include <cstdio>
#include <cstdlib>

int main()
{
    clock_t start, end;
    float elapsed1, elapsed2, elapsed3;

    /* 初始化 */
    std::fprintf(stdout, "initializing...\r\n");
    DATA1 = (float*)std::malloc(XS * YS * sizeof(float));
    DATA2 = (float*)std::malloc(XS * YS * sizeof(float));
    initialize(DATA1, def_val);
    initialize(DATA2, def_val);

    /* 执行串行内核 */
    std::fprintf(stdout, "computing 1...\r\n");
    start = clock();
    kernel_serial(DATA1);
    end = clock();
    elapsed1 = (float)(end - start) / CLK_TCK;
    std::fprintf(stdout, "elapsed1 = %f s\r\n", elapsed1);

    /* 执行波阵面并行内核 */
    std::fprintf(stdout, "computing 2...\r\n");
    start = clock();
    kernel_skew(DATA2);
    end = clock();
    elapsed2 = (float)(end - start) / CLK_TCK;
    if (verify(DATA1, DATA2))
        std::fprintf(stdout, "elapsed2 = %f s\r\n", elapsed2);
    else
        std::fprintf(stderr, "error 2! elapsed2 = %f s\r\n", elapsed2);

    std::free(DATA2);
    std::free(DATA1);
}