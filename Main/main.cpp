#include <iostream>
#include "FrequencyPredictingModel.hpp"
#include "simple_file_parser.hpp"

using namespace orfp;

int main()
{
	std::string modelPath = "..\\onnx_models\\lstm_ffnn_shift_version.onnx";
	const int64_t seqLen = 8;
	const float default_shift = 0.0f;

	auto model = FrequencyPredictingModel(modelPath, seqLen, default_shift);

	std::ifstream timeFile("..\\test_input\\time.txt");
	std::ifstream temperatureFile("..\\test_input\\temperature.txt");

	try
	{
		auto timeSeries = _impl::parse(timeFile);
		auto temperatureSeries = _impl::parse(temperatureFile);

		for (size_t i = 0; i < timeSeries.size(); ++i)
		{
			float freq = model.predict(timeSeries[i], temperatureSeries[i]);
			std::cout << "Prediction at t = " << timeSeries[i] << " and T = " << temperatureSeries[i] << ": " << freq << std::endl;
		}

	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	timeFile.close();
	temperatureFile.close();

	return 0;
}
