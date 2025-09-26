#include "utils.hpp"

namespace
{
    static float scalar_product(const float* x, const float* y, size_t length)
    {
        float product = 0.0f;
        for (size_t i = 0; i < length; ++i)
        {
            product += x[i] * y[i];
        }

        return product;
    }

    static float sum(const float* x, size_t length)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < length; ++i)
        {
            sum += x[i];
        }

        return sum;
    }
}

namespace orfp
{
    namespace _impl
    {
        float compute_derivative_by_linear_fit(const float* x, const float* y, size_t length)
        {
            float x_sum = sum(x, length);
            float y_sum = sum(y, length);

            float derivative = (length * scalar_product(x, y, length) - x_sum * y_sum) / (length * scalar_product(x, x, length) - x_sum * x_sum);

            return derivative;
        }
    }
}