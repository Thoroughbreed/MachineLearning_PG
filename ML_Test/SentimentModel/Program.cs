using System;

namespace SentimentModel.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        { Console.WriteLine("=============== Press the any key to try the sentence-o-meter ===============");
            Console.ReadKey();

            string input;
            do
            {
                Console.Write("Input data: ");
                input = Console.ReadLine();
                var sampleData = new SentimentModel.ModelInput() { Col0 = input };
                var modelOutput = SentimentModel.Predict(sampleData);
                bool posiometer = modelOutput.PredictedLabel == 0;
                Console.WriteLine($"Positive: {posiometer}, {(modelOutput.Score[0]*100).ToString("N2")}% negative" +
                                  $" - {(modelOutput.Score[1]*100).ToString("N2")}% positive");
            } while (input != "iddqd");
        }
    }
}
