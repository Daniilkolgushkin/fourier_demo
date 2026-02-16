#include <emscripten/emscripten.h>
#include <cmath>
#include <complex>
#include <vector>

const float PI = 3.141592653589793f;

static unsigned int bit_reverse(unsigned int x, int log2n) {
    unsigned int n = 0;
    for (int i = 0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

// Fast Fourier Transform (inverse = true – inverse otherwise direct)
static void fft(float* real, float* imag, int n, bool is_inverse) {
    int log2n = 0;
    int k = n;
    while (k >>= 1) ++log2n;

    std::vector<std::complex<float>> temp(n);
    for (int i = 0; i < n; ++i) {
        int rev = bit_reverse(i, log2n);
        temp[rev] = std::complex<float>(real[i], imag[i]);
    }

    for (int s = 1; s <= log2n; ++s) {
        int m = 1 << s;
        int m2 = m >> 1;
        std::complex<float> w(1.0f, 0.0f);
        float angle = (is_inverse ? 2.0f : -2.0f) * PI / m;
        std::complex<float> wm = std::polar(1.0f, angle);
        for (int j = 0; j < m2; ++j) {
            for (int i = j; i < n; i += m) {
                int k = i + m2;
                std::complex<float> u = temp[i];
                std::complex<float> v = temp[k] * w;
                temp[i] = u + v;
                temp[k] = u - v;
            }
            w *= wm;
        }
    }

    if (is_inverse) {
        for (int i = 0; i < n; ++i) {
            real[i] = temp[i].real() / n;
            imag[i] = temp[i].imag() / n;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            real[i] = temp[i].real();
            imag[i] = temp[i].imag();
        }
    }
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
void forward_fft(float* real, float* imag, int n) {
    fft(real, imag, n, false);
}

EMSCRIPTEN_KEEPALIVE
void inverse_fft(float* real, float* imag, int n) {
    fft(real, imag, n, true);
}

EMSCRIPTEN_KEEPALIVE
void modify_spectrum(float* real, float* imag, int n, int k_min, int k_max, int remove_phase) {
    for (int k = 0; k < n; ++k) {
        if (k < k_min || k > k_max) {
            real[k] = 0.0f;
            imag[k] = 0.0f;
        }
    }
    if (remove_phase) {
        int half = n / 2;
        for (int k = 0; k <= half; ++k) {
            int k_sym = (k == 0 || k == half) ? k : n - k;
            float mag = std::sqrt(real[k] * real[k] + imag[k] * imag[k]);
            real[k] = mag;
            imag[k] = 0.0f;
            if (k_sym != k) {
                real[k_sym] = mag;
                imag[k_sym] = 0.0f;
            }
        }
    }
}

} // extern "C"
