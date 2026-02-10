#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

#ifdef USE_FLOAT
    using TypeDig = float;
#else
    using TypeDig = double;
#endif

int main() {
    const int N = 10000000;
    const TypeDig STEP = 2.0 * M_PI / N;
    std::vector<TypeDig>arr(N);

    for (int i = 0; i < N; i++) {
        arr[i] = std::sin(i * STEP);
    }

    std::cout << "Сумма значений синуса: " << std::accumulate(arr.begin(), arr.end(), 0.0) << '\n';
    return 0;
}