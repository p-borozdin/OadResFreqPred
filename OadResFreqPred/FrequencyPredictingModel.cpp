#include <onnxruntime_cxx_api.h>
#include "utils.hpp"
#include "FrequencyPredictingModel.hpp"

using namespace orfp;

namespace
{
	const float TEMPERATURE_MEAN = 25.959616468039005f;
	const float TEMPERATURE_STD = 4.876389994267977f;

	const float TEMPERATURE_GRADIENT_MEAN = -0.0005538732771624923f;
	const float TEMPERATURE_GRADIENT_STD = 0.010795512399269446f;

	inline float normalize(float value, float valueMean, float valueStd)
	{
		return (value - valueMean) / valueStd;
	}

	inline float restoreOutput(float output)
	{
		return output * 1000.0f;
	}

	const wchar_t* stringToWcharPtr(const std::string& string)
	{
		size_t requiredSize = 0;
		mbstowcs_s(&requiredSize, nullptr, 0, string.c_str(), _TRUNCATE);

		if (requiredSize == 0)
		{
			return nullptr;
		}

		wchar_t* wideCharPtr = new wchar_t[requiredSize];
		mbstowcs_s(&requiredSize, wideCharPtr, requiredSize, string.c_str(), _TRUNCATE);
		return wideCharPtr;
	}
}

FrequencyPredictingModel::FrequencyPredictingModel(
	const std::string& modelPath,
	int64_t seqLen,
	float default_shift) :

	m_modelPath(stringToWcharPtr(modelPath)),
	m_seqLen(seqLen),
	m_defaultShift(default_shift),
	m_inputShape({1, seqLen, 2}),
	m_session(Ort::Env(), m_modelPath, Ort::SessionOptions()),
	m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),

	m_timeBuffer(RAW_BUFFER_SIZE),
	m_temperatureBuffer(RAW_BUFFER_SIZE),
	m_inputBuffer(seqLen)
{
	if (seqLen <= 0)
	{
		throw std::runtime_error("Parameter 'seqLen' must be greater than 0");
	}
}

float FrequencyPredictingModel::interpolateTemperatureAt(float target_time) const
{
	if (!m_timeBuffer.isFilled() || !m_temperatureBuffer.isFilled())
	{
		return 0.0f;
	}
	
	const float* raw_times = m_timeBuffer.data();
	const float* raw_temps = m_temperatureBuffer.data();

	for (size_t j = 0; j < RAW_BUFFER_SIZE - 1; ++j)
	{
		if (raw_times[j] <= target_time && raw_times[j + 1] >= target_time)
		{
			float t1 = raw_times[j], t2 = raw_times[j + 1];
			float T1 = raw_temps[j], T2 = raw_temps[j + 1];
			
			if (std::abs(t2 - t1) > 1e-6f)
			{
				float alpha = (target_time - t1) / (t2 - t1);
				return T1 + alpha * (T2 - T1);
			}
			else
			{
				return T1;
			}
		}
	}
	
	if (target_time < raw_times[0])
	{
		return raw_temps[0];
	}
	else if (target_time > raw_times[RAW_BUFFER_SIZE - 1])
	{
		return raw_temps[RAW_BUFFER_SIZE - 1];
	}
	
	return 0.0f;
}

float FrequencyPredictingModel::computeGradientAt(float center_time) const
{	
	std::vector<float> window_times;
	std::vector<float> window_temps;
	window_times.reserve(GRADIENT_WINDOW_SIZE);
	window_temps.reserve(GRADIENT_WINDOW_SIZE);
	
	for (size_t i = 0; i < GRADIENT_WINDOW_SIZE; ++i)
	{
		float t = center_time - (GRADIENT_WINDOW_SIZE - 1 - i) * INTERPOLATION_INTERVAL;
		float T = interpolateTemperatureAt(t);
		
		window_times.push_back(t);
		window_temps.push_back(T);
	}
	
	// Вычисляем градиент по методу наименьших квадратов
	return _impl::compute_derivative_by_linear_fit(
		window_times.data(),
		window_temps.data(),
		GRADIENT_WINDOW_SIZE
	);
}

float FrequencyPredictingModel::predict(float time, float temperature)
{
	m_timeBuffer.add(time);
	m_temperatureBuffer.add(temperature);

	if (!m_timeBuffer.isFilled() || !m_temperatureBuffer.isFilled())
	{
		return 0.0f;
	}

	const float last_time = m_timeBuffer.data()[RAW_BUFFER_SIZE - 1];

	for (size_t i = 0; i < m_seqLen; ++i)
	{
		float target_time = last_time - (m_seqLen - 1 - i) * INTERPOLATION_INTERVAL;
		
		float temperature_raw = interpolateTemperatureAt(target_time);
		
		float temperatureGrad = computeGradientAt(target_time);
		
		float temperature_shifted = temperature_raw + m_defaultShift;
		float temperatureNormalized = normalize(temperature_shifted, TEMPERATURE_MEAN, TEMPERATURE_STD);
		float temperatureGradNormalized = normalize(temperatureGrad, TEMPERATURE_GRADIENT_MEAN, TEMPERATURE_GRADIENT_STD);
		
		m_inputBuffer.add(temperatureNormalized, temperatureGradNormalized);
	}

	if (!m_inputBuffer.isFilled()) 
	{
		return 0.0f;
	}

	return predictInternal();
}

float FrequencyPredictingModel::predictInternal() 
{
	float* pInputData = reinterpret_cast<float*>(m_inputBuffer.data());
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
		m_memoryInfo,
		pInputData,
		2 * m_seqLen,
		m_inputShape.data(),
		m_inputShape.size()
	);

	std::vector<const char*> inputName = { "input" };
	std::vector<const char*> outputName = { "output" };

	auto outputTensors = m_session.Run(
		Ort::RunOptions{ nullptr },
		inputName.data(), &inputTensor, 1,
		outputName.data(), 1
	);
	const float* pOutputData = outputTensors[0].GetTensorData<float>();

	return restoreOutput(*pOutputData);
}

float FrequencyPredictingModel::getShift() const
{
    return m_defaultShift;
}
