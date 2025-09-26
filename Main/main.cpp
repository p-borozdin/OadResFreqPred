#include <iostream>
#include "FrequencyPredictingModel.hpp"
#include "simple_file_parser.hpp"

using namespace orfp;

int main()
{
	std::string modelPath = "..\\onnx_models\\lstm_sa.onnx";
	const int64_t seqLen = 7;

	auto model = FrequencyPredictingModel(modelPath, seqLen);

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
