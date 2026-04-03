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

	const size_t WINDOW_SIZE = 15;

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

FrequencyPredictingModel::FrequencyPredictingModel(const std::string& modelPath, int64_t seqLen, float default_shift) :
	m_modelPath(stringToWcharPtr(modelPath)),
	m_seqLen(seqLen),
	m_defaultShift(default_shift),
	m_inputShape({1, seqLen, 2}),
	m_session(Ort::Env(), m_modelPath, Ort::SessionOptions()),
	m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
	m_timeBuffer(WINDOW_SIZE),
	m_temperatureBuffer(WINDOW_SIZE),
	m_inputBuffer(seqLen)
{
	if (seqLen <= 0)
	{
		throw std::exception("Parameter 'seqLen' must be greater than 0");
	}
}

float FrequencyPredictingModel::predict(float time, float temperature)
{
	m_timeBuffer.add(time);
	m_temperatureBuffer.add(temperature);

	if (!m_timeBuffer.isFilled() || !m_temperatureBuffer.isFilled())
	{
		return 0.0f;
	}

	float temperatureGrad = _impl::compute_derivative_by_linear_fit(
		m_timeBuffer.data(),
		m_temperatureBuffer.data(),
		WINDOW_SIZE
	);

	float temperature_shifted = temperature + m_defaultShift;

	float temperatureNormalized = normalize(temperature_shifted, TEMPERATURE_MEAN, TEMPERATURE_STD);
	float temperatureGradNormalized = normalize(temperatureGrad, TEMPERATURE_GRADIENT_MEAN, TEMPERATURE_GRADIENT_STD);

	m_inputBuffer.add(temperatureNormalized, temperatureGradNormalized);

	if (!m_inputBuffer.isFilled()) 
	{
		return 0.0f;
	}

	return predictInternal();
}

float FrequencyPredictingModel::predictInternal() 
{
	float* pInputData = reinterpret_cast<float*>(m_inputBuffer.data());
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(m_memoryInfo, pInputData, 2 * m_seqLen, m_inputShape.data(), m_inputShape.size());

	std::vector<const char*> inputName = { "input" };
	std::vector<const char*> outputName = { "output" };

	auto outputTensors = m_session.Run(Ort::RunOptions{ nullptr }, inputName.data(), &inputTensor, 1, outputName.data(), 1);
	const float* pOutputData = outputTensors[0].GetTensorData<float>();

	return restoreOutput(*pOutputData);
}

float FrequencyPredictingModel::getShift() const
{
    return m_defaultShift;
}