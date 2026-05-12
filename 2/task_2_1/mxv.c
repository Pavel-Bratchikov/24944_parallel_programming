#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static const int THREADS[] = {1, 2, 4, 7, 8, 16, 20, 40};
static const int THREADS_COUNT = 8;

double fetch_time() {
    struct timespec t_spec;
    timespec_get(&t_spec, TIME_UTC);
    return (double)t_spec.tv_sec + (double)t_spec.tv_nsec * 1e-9;
}

void calc_mv(double *mat, double *vec, double *res, size_t rows, size_t cols) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            sum += mat[r * cols + c] * vec[c];
        }
        res[r] = sum;
    }
}

double run_serial(double *mat, double *vec, double *res, size_t rows, size_t cols) {
    double t0 = fetch_time();

    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            sum += mat[r * cols + c] * vec[c];
        }
        res[r] = sum;
    }

    return fetch_time() - t0;
}

double run_parallel(double *mat, double *vec, double *res,
                     size_t rows, size_t cols, int threads) {

    omp_set_num_threads(threads);

    double t0 = fetch_time();

#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            sum += mat[r * cols + c] * vec[c];
        }
        res[r] = sum;
    }

    return fetch_time() - t0;
}

void run_case(size_t rows, size_t cols) {
    printf("\n==============================\n");
    printf("Matrix: %zu x %zu\n", rows, cols);
    printf("==============================\n");

    double *mat = malloc(sizeof(double) * rows * cols);
    double *vec = malloc(sizeof(double) * cols);
    double *res = malloc(sizeof(double) * rows);

    if (!mat || !vec || !res) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    // init
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            mat[r * cols + c] = (double)(r + c);

    for (size_t c = 0; c < cols; ++c)
        vec[c] = (double)c;

    // SERIAL
    double t_serial = run_serial(mat, vec, res, rows, cols);
    printf("[Serial]   time: %.6f s\n", t_serial);

    // PARALLEL VARIANTS
    for (int i = 0; i < THREADS_COUNT; ++i) {
        int t = THREADS[i];

        double t_par = run_parallel(mat, vec, res, rows, cols, t);
        printf("[OMP %2d]   time: %.6f s | speedup: %.3fx\n",
               t, t_par, t_serial / t_par);
    }

    free(mat);
    free(vec);
    free(res);
}

int main() {
    size_t sizes[][2] = {
        {20000, 20000},
        {40000, 40000}
    };

    int cases = 2;

    printf("OpenMP benchmark: matrix-vector multiplication\n");

    for (int i = 0; i < cases; ++i) {
        run_case(sizes[i][0], sizes[i][1]);
    }

    return 0;
}