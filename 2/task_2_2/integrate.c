#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

const double PI_CONST = 3.14159265358979323846;
const double limit_a = -4.0;
const double limit_b = 4.0;
const int total_steps = 40000000;

int THREADS_LIST[] = {1, 2, 4, 7, 8, 16, 20, 40};
int THREADS_COUNT = 8;

double get_runtime() {
    struct timespec t_spec;
    timespec_get(&t_spec, TIME_UTC);
    return (double)t_spec.tv_sec + (double)t_spec.tv_nsec * 1.e-9;
}

double math_function(double val) {
    return exp(-val * val);
}

double compute_integral_omp(double (*func_ptr)(double),
                             double start,
                             double end,
                             int steps,
                             int num_threads) {

    double step_size = (end - start) / steps;
    double accumulated_sum = 0.0;

    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        int num_t = omp_get_num_threads();
        int t_id = omp_get_thread_num();

        int chunk = steps / num_t;
        int lower_bound = t_id * chunk;
        int upper_bound = (t_id == num_t - 1) ? (steps - 1)
                                               : (lower_bound + chunk - 1);

        double local_sum = 0.0;

        for (int k = lower_bound; k <= upper_bound; ++k) {
            local_sum += func_ptr(start + step_size * (k + 0.5));
        }

#pragma omp atomic
        accumulated_sum += local_sum;
    }

    return accumulated_sum * step_size;
}

double compute_serial(double (*func_ptr)(double),
                      double start,
                      double end,
                      int steps) {

    double step_size = (end - start) / steps;
    double sum = 0.0;

    for (int k = 0; k < steps; ++k) {
        sum += func_ptr(start + step_size * (k + 0.5));
    }

    return sum * step_size;
}

int main() {

    printf("--- Integral exp(-x^2) benchmark ---\n");
    printf("Range: [%.1f, %.1f], steps = %d\n\n",
           limit_a, limit_b, total_steps);

    // SERIAL (один раз)
    double t0 = get_runtime();
    double serial_result =
        compute_serial(math_function, limit_a, limit_b, total_steps);
    double serial_time = get_runtime() - t0;

    printf("[Serial]   result = %.12f | time = %.6f s\n\n",
           serial_result, serial_time);

    // PARALLEL VARIANTS
    for (int i = 0; i < THREADS_COUNT; i++) {

        int threads = THREADS_LIST[i];

        double t1 = get_runtime();
        double par_result =
            compute_integral_omp(math_function,
                                  limit_a,
                                  limit_b,
                                  total_steps,
                                  threads);

        double par_time = get_runtime() - t1;

        printf("[OMP %2d]   time = %.6f s | speedup = %.3fx | error = %.12f\n",
               threads,
               par_time,
               serial_time / par_time,
               fabs(par_result - sqrt(PI_CONST)));
    }

    return 0;
}