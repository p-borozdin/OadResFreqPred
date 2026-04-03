#pragma once
#include <string>
#include <vector>

namespace orfp
{
	/// <summary>
	/// Результат калибровки температурного сдвига.
	/// </summary>
	struct CalibrationResult final {
		float optimal_shift{0.0f};      // Оптимальный сдвиг температуры (°C)
		float min_mae{0.0f};            // Минимальная MAE (Гц)
		size_t total_points{0};         // Всего точек данных
		size_t valid_points{0};         // Точек использовано для MAE (после warmup)
		size_t warmup_points{0};        // Точек пропущено (warmup период)
		float shift_search_range_min{-5.0f};  // Диапазон поиска сдвига
		float shift_search_range_max{5.0f};
		float shift_search_step{0.01f};        // Шаг поиска сдвига
	};

	/// <summary>
	/// Класс для поиска оптимального температурного сдвига по метрике MAE.
	/// </summary>
	class TemperatureShiftCalibrator final {
	public:
		/// <summary>
		/// Конструктор.
		/// </summary>
		/// <param name="modelPath">Путь к ONNX модели.</param>
		/// <param name="seqLen">Длина последовательности LSTM.</param>
		TemperatureShiftCalibrator(const std::string& modelPath, int64_t seqLen);

		/// <summary>
		/// Калибровка температурного сдвига.
		/// </summary>
		/// <param name="timeFile">Путь к файлу времени (time.txt).</param>
		/// <param name="tempFile">Путь к файлу температуры (temperature.txt).</param>
		/// <param name="freqFile">Путь к файлу целевой частоты (frequency.txt).</param>
		/// <param name="warmupPoints">Количество начальных точек для пропуска (не учитывать в MAE).</param>
		/// <param name="shiftMin">Минимальный сдвиг для поиска (°C).</param>
		/// <param name="shiftMax">Максимальный сдвиг для поиска (°C).</param>
		/// <param name="shiftStep">Шаг поиска сдвига (°C).</param>
		/// <returns>Результат калибровки с оптимальным сдвигом и MAE.</returns>
		CalibrationResult calibrate(
			const std::string& timeFile,
			const std::string& tempFile,
			const std::string& freqFile,
			size_t warmupPoints = 50,
			float shiftMin = -5.0f,
			float shiftMax = 5.0f,
			float shiftStep = 0.01f
		);

		/// <summary>
		/// Получить последний результат калибровки.
		/// </summary>
		const CalibrationResult& getLastResult() const { return m_lastResult; }

	private:
		std::string m_modelPath;
		int64_t m_seqLen;
		CalibrationResult m_lastResult;

		/// <summary>
		/// Загрузить данные из текстового файла (одна колонка float).
		/// </summary>
		std::vector<float> loadData(const std::string& filepath) const;

		/// <summary>
		/// Вычислить MAE между предсказаниями и целевой частотой.
		/// </summary>
		float computeMAE(
			const std::vector<float>& predictions,
			const std::vector<float>& targets,
			size_t skipFirst
		) const;
	};
}