#pragma once
#include <onnxruntime_cxx_api.h>
#include "LimitedBuffer.hpp"

namespace orfp
{
	class FrequencyPredictingModel final {
	public:
		/// <summary>
		/// Создает модель для предсказания резонансной частоты.
		/// </summary>
		/// <param name="modelPath">-- путь до *.onnx файла с моделью. </param>
		/// <param name="seqLen">-- гиперпараметр seq_len. </param>
		FrequencyPredictingModel(const wchar_t* modelPath, int64_t seqLen);

		/// <summary>
		/// Делает предсказание для резонансной частоты.
		/// </summary>
		/// <param name="time">-- отметка времени. </param>
		/// <param name="temperature">-- температура в момент времени time. </param>
		/// <returns> Предсказанная частота. Обратите внимание, что 
		/// (1) сначала модель будет заполнять внутренний буфер для подсчета градиента dT/dt; 
		/// (2) затем модель будет заполнять внутренний буфер для LSTM ячейки; 
		/// и только после этого она будет выдавать предсказания. Пока буферы не будут заполнены, модель будет выдавать значение 0. </returns>
		float predict(float time, float temperature);

	private:
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