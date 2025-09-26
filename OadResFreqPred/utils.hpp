#pragma once

namespace orfp // OadResFreqPred namespace
{
	namespace _impl // internal implementation
	{
		// Интерполирует y(x) прямой методом наименьших квадратов и возвращает коэффициент при х.
		// y = ax + b => y' = a
		float compute_derivative_by_linear_fit(const float* x, const float* y, size_t length);
	}
}
