#pragma once
#include <onnxruntime_cxx_api.h>
#include "LimitedBuffer.hpp"

namespace orfp
{
	class FrequencyPredictingModel final {
	public:
		/// <summary>
		/// Creates model for resonance frequency prediction.
		/// </summary>
		/// <param name="modelPath">-- path to *.onnx file with a model. </param>
		/// <param name="seqLen">-- hyperparameters seq_len. </param>
		FrequencyPredictingModel(const std::string& modelPath, int64_t seqLen);

		~FrequencyPredictingModel() { delete m_modelPath; }

		/// <summary>
		/// Predicts resonance frequency.
		/// </summary>
		/// <param name="time">-- timestamp in seconds. </param>
		/// <param name="temperature">-- temperature in Celsius at the given time. </param>
		/// <returns> Predicted frequency. Note that
		/// (1) at first, the buffer for the temperature gradient (dT/dt) will be filling;
		/// (2) then, the internal LSTM buffer will be filling; 
		/// and only after that the model will give predicted frequency. Until the buffers are filled, the model will return 0. </returns>
		float predict(float time, float temperature);

	private:
		const wchar_t* m_modelPath{ nullptr };
		size_t m_seqLen{};
		std::vector<int64_t> m_inputShape{};
		Ort::Session m_session;
		Ort::MemoryInfo m_memoryInfo;

		LimitedBuffer<float> m_timeBuffer;
		LimitedBuffer<float> m_temperatureBuffer;

		struct model_input final {
			float temperature{};
			float temperatureGradient{};
		};

		LimitedBuffer<model_input> m_inputBuffer;

		float predictInternal();
	};
}