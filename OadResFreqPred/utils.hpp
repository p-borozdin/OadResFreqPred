#pragma once

namespace orfp // OadResFreqPred namespace
{
	namespace _impl // internal implementation
	{
		// Interpolate y(x) by linear function with the least squares method and return the coefficient before x:
		// y = ax + b => y' = a
		float compute_derivative_by_linear_fit(const float* x, const float* y, size_t length);
	}
}
