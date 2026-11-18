#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <complex>
#include <vector>

using C = std::complex<double>;
#if defined(_WIN32)
#define ITERATED_EXPORT __declspec(dllexport)
#else
#define ITERATED_EXPORT __attribute__((visibility("default")))
#endif

template <int StaticStride>
void evaluate(
    int m, int letters, int words, int outputs, int level,
    const int* letter_k, const int* letter_r, const double* time_poly,
    const int* grade, const int* parent, const int* last_letter,
    const int* split_count, const int* splits, const int* output_words,
    const int* grad_starts, const int* grad_splits,
    const C* q, C* result, C* jacobian
) {
    const int stride = StaticStride > 0 ? StaticStride : level + 1;
    const bool with_jacobian = jacobian != nullptr;
    std::vector<C> local(m * words), local_deriv(with_jacobian ? m * words : 0);
    std::vector<C> ints(words * stride), ints_deriv(with_jacobian ? words * stride : 0);
    std::vector<C> generator(letters * stride), gen_deriv(with_jacobian ? letters * stride : 0);
    ints[0] = 1.0;
    const double step = 1.0 / m;
    std::vector<double> inv(stride);
    for (int i = 1; i < stride; ++i) inv[i] = 1.0 / i;
    // Form coefficients of (q[j] + 1/2 - u)^k and multiply by the
    // precomputed local Legendre polynomial for each generator.
    std::vector<C> powers(stride * stride);
    for (int j = 0; j < m; ++j) {
        std::fill(powers.begin(), powers.end(), C{});
        powers[0] = 1.0;
        const C value = q[j] + 0.5;
        for (int k = 1; k < level; ++k) {
            powers[k * stride] = value * powers[(k - 1) * stride];
            for (int p = 1; p < k; ++p)
                powers[k * stride + p] = value * powers[(k - 1) * stride + p] - powers[(k - 1) * stride + p - 1];
            powers[k * stride + k] = -powers[(k - 1) * stride + k - 1];
        }
        std::fill(generator.begin(), generator.end(), C(0));
        std::fill(gen_deriv.begin(), gen_deriv.end(), C(0));
        for (int l = 0; l < letters; ++l) {
            const int k = letter_k[l], r = letter_r[l];
            const double* t = time_poly + (j * letters + l) * stride;
            C* g = generator.data() + l * stride;
            for (int a = 0; a <= r; ++a)
                for (int b = 0; b <= k; ++b)
                    g[a + b] += t[a] * powers[k * stride + b];
            if (with_jacobian) {
                C* dg = gen_deriv.data() + l * stride;
                for (int a = 0; a <= r; ++a)
                    for (int b = 0; b < k; ++b)
                        dg[a + b] += (t[a] * k) * powers[(k - 1) * stride + b];
            }
        }
        // An iterated integral extends a prefix by one generator.
        // Integrate each polynomial term analytically over local time u.
        local[j * words] = 1.0;
        for (int w = 1; w < words; ++w) {
            const int p = parent[w], l = last_letter[w];
            const int prefix_degree = grade[p], letter_degree = grade[w] - prefix_degree - 1;
            const C* f = ints.data() + p * stride;
            const C* g = generator.data() + l * stride;
            C* product = ints.data() + w * stride;
            std::fill(product, product + grade[w] + 1, C(0));
            for (int a = 0; a <= prefix_degree; ++a)
                for (int b = 0; b <= letter_degree; ++b)
                    product[a + b + 1] += f[a] * g[b];
            C evaluated = 0.0;
            for (int k = 1; k <= grade[w]; ++k) {
                product[k] *= step * inv[k];
                evaluated += product[k];
            }
            local[j * words + w] = evaluated;
            if (with_jacobian) {
                const C* df = ints_deriv.data() + p * stride;
                const C* dg = gen_deriv.data() + l * stride;
                C* dproduct = ints_deriv.data() + w * stride;
                std::fill(dproduct, dproduct + grade[w] + 1, C(0));
                for (int a = 0; a <= prefix_degree; ++a)
                    for (int b = 0; b <= letter_degree; ++b)
                        dproduct[a + b + 1] += f[a] * dg[b] + df[a] * g[b];
                C derivative = 0.0;
                for (int k = 1; k <= grade[w]; ++k) {
                    dproduct[k] *= step * inv[k];
                    derivative += dproduct[k];
                }
                local_deriv[j * words + w] = derivative;
            }
        }
    }

    // Concatenate intervals by splitting each word at every possible cut.
    std::vector<C> before((m + 1) * words);
    before[0] = 1.0;
    for (int j = 0; j < m; ++j) {
        const C* left = before.data() + j * words;
        const C* right = local.data() + j * words;
        C* out = before.data() + (j + 1) * words;
        out[0] = 1.0;
        for (int w = 1; w < words; ++w) {
            C sum = 0.0;
            const int* index = splits + 2 * split_count[w];
            for (int s = split_count[w]; s < split_count[w + 1]; ++s, index += 2)
                sum += left[index[0]] * right[index[1]];
            out[w] = sum;
        }
    }
    for (int i = 0; i < outputs; ++i)
        result[i] = before[m * words + output_words[i]];
    if (!with_jacobian) return;

    // A derivative of the full product uses the interval derivative between
    // the concatenated prefixes and suffixes on either side.
    std::vector<C> after((m + 1) * words);
    after[m * words] = 1.0;
    for (int j = m - 1; j >= 0; --j) {
        const C* left = local.data() + j * words;
        const C* right = after.data() + (j + 1) * words;
        C* out = after.data() + j * words;
        out[0] = 1.0;
        for (int w = 1; w < words; ++w) {
            C sum = 0.0;
            const int* index = splits + 2 * split_count[w];
            for (int s = split_count[w]; s < split_count[w + 1]; ++s, index += 2)
                sum += left[index[0]] * right[index[1]];
            out[w] = sum;
        }
    }
    for (int i = 0; i < outputs; ++i) {
        for (int j = 0; j < m; ++j) {
            const C* left = before.data() + j * words;
            const C* mid = local_deriv.data() + j * words;
            const C* right = after.data() + (j + 1) * words;
            C sum = 0.0;
            const int* index = grad_splits + 3 * grad_starts[i];
            for (int s = grad_starts[i]; s < grad_starts[i + 1]; ++s, index += 3)
                sum += left[index[0]] * mid[index[1]] * right[index[2]];
            jacobian[i * m + j] = sum;
        }
    }
}

// ctypes gets this symbol from the installed extension. The Python caller
// owns the validated contiguous arrays and keeps them alive during the call.
extern "C" ITERATED_EXPORT void iterated_eval(
    int m, int letters, int words, int outputs, int level,
    const int* letter_k, const int* letter_r, const double* time_poly,
    const int* grade, const int* parent, const int* last_letter,
    const int* split_count, const int* splits, const int* output_words,
    const int* grad_starts, const int* grad_splits,
    const C* q, C* result, C* jacobian
) {
    // Keep the short, fixed-stride path of the original kernel.
    auto kernel = level < 10 ? evaluate<10> : evaluate<0>;
    kernel(m, letters, words, outputs, level, letter_k, letter_r, time_poly,
           grade, parent, last_letter, split_count, splits, output_words,
           grad_starts, grad_splits, q, result, jacobian);
}

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_iterated", nullptr, -1, nullptr
};

PyMODINIT_FUNC PyInit__iterated() {
    return PyModule_Create(&module);
}
