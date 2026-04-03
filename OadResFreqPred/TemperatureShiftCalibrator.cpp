#include "TemperatureShiftCalibrator.hpp"
#include "FrequencyPredictingModel.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

namespace orfp
{
	TemperatureShiftCalibrator::TemperatureShiftCalibrator(
		const std::string& modelPath, 
		int64_t seqLen) :
		m_modelPath(modelPath),
		m_seqLen(seqLen)
	{
	}

	std::vector<float> TemperatureShiftCalibrator::loadData(const std::string& filepath) const
	{
		std::vector<float> values;
		std::ifstream file(filepath);
		
		if (!file.is_open())
		{
			throw std::runtime_error("Cannot open file: " + filepath);
		}

		std::string line;
		while (std::getline(file, line))
		{
			if (line.empty()) continue;
			
			try
			{
				float value = std::stof(line);
				values.push_back(value);
			}
			catch (const std::exception& e)
			{
				std::cerr << "Warning: Skipping invalid line in " << filepath << ": " << line << std::endl;
			}
		}

		file.close();
		return values;
	}

	float TemperatureShiftCalibrator::computeMAE(
		const std::vector<float>& predictions,
		const std::vector<float>& targets,
		size_t skipFirst
	) const
	{
		if (predictions.size() != targets.size())
		{
			throw std::runtime_error(
				"Predictions (" + std::to_string(predictions.size()) + 
				") and targets (" + std::to_string(targets.size()) + ") size mismatch"
			);
		}

		if (skipFirst >= predictions.size())
		{
			return std::numeric_limits<float>::max();
		}

		float sum_abs_error = 0.0f;
		size_t count = 0;

		for (size_t i = skipFirst; i < predictions.size(); ++i)
		{
			// Пропускать нулевые предсказания (период прогрева буферов)
			if (predictions[i] == 0.0f) continue;

			sum_abs_error += std::abs(predictions[i] - targets[i]);
			count++;
		}

		if (count == 0)
		{
			return std::numeric_limits<float>::max();
		}

		return sum_abs_error / static_cast<float>(count);
	}

	CalibrationResult TemperatureShiftCalibrator::calibrate(
		const std::string& timeFile,
		const std::string& tempFile,
		const std::string& freqFile,
		size_t warmupPoints,
		float shiftMin,
		float shiftMax,
		float shiftStep
	)
	{
		auto times = loadData(timeFile);
		auto temps = loadData(tempFile);
		auto target_freqs = loadData(freqFile);

		// Проверка согласованности размеров
		if (times.size() != temps.size())
		{
			throw std::runtime_error(
				"Time (" + std::to_string(times.size()) + 
				") and temperature (" + std::to_string(temps.size()) + ") size mismatch"
			);
		}

		if (times.size() != target_freqs.size())
		{
			throw std::runtime_error(
				"Time (" + std::to_string(times.size()) + 
				") and frequency (" + std::to_string(target_freqs.size()) + ") size mismatch"
			);
		}

		// 2️⃣ Поиск оптимального сдвига
		CalibrationResult result;
		result.optimal_shift = 0.0f;
		result.min_mae = std::numeric_limits<float>::max();
		result.total_points = times.size();
		result.warmup_points = warmupPoints;
		result.shift_search_range_min = shiftMin;
		result.shift_search_range_max = shiftMax;
		result.shift_search_step = shiftStep;

		for (float shift = shiftMin; shift <= shiftMax + 1e-6f; shift += shiftStep)
		{
			// Создаём новую модель для каждого сдвига (чистое состояние буферов)
			FrequencyPredictingModel model(m_modelPath, m_seqLen, shift);

			std::vector<float> predictions;
			predictions.reserve(times.size());

			// Прогон через все точки
			for (size_t i = 0; i < times.size(); ++i)
			{
				float freq = model.predict(times[i], temps[i]);
				predictions.push_back(freq);
			}

			// Вычисление MAE (пропускаем warmup точки и нулевые предсказания)
			float mae = computeMAE(predictions, target_freqs, warmupPoints);

			if (mae < result.min_mae)
			{
				result.min_mae = mae;
				result.optimal_shift = shift;
			}
		}

		// Подсчёт валидных точек
		size_t valid_count = 0;
		for (size_t i = warmupPoints; i < predictions.size(); ++i)
		{
			if (predictions[i] != 0.0f) valid_count++;
		}
		result.valid_points = valid_count;

		m_lastResult = result;
		return result;
	}
}
