using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;

namespace ML_Test
{
    class Program
    {
        private static IDataView predictImageData;
        static void Main(string[] args)
        {
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
            var assetsRelativePath = Path.Combine(projectDirectory, "assets");
            var imageToPredictPath = Path.Combine(projectDirectory, "predict");

            MLContext mlContext = new MLContext();

            IEnumerable<ImageData> images =
                LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
            IEnumerable<ImageData> imagesToPredict =
                LoadImagesFromDirectory(folder: imageToPredictPath, useFolderNameAsLabel: true);

            IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
            predictImageData = mlContext.Data.LoadFromEnumerable(imagesToPredict);   
            
            var preprocessingPredictionPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Label",
                    outputColumnName: "LabelAsKey")
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: assetsRelativePath,
                    inputColumnName: "ImagePath"));
            
            IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);
            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Label",
                    outputColumnName: "LabelAsKey")
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: assetsRelativePath,
                    inputColumnName: "ImagePath"));

            IDataView preProcessedData = preprocessingPipeline
                .Fit(shuffledData)
                .Transform(shuffledData);
            IDataView preProcessedPredictionData = preprocessingPredictionPipeline.Fit(predictImageData).Transform(predictImageData);
            TrainTestData predictionTest =
                mlContext.Data.TrainTestSplit(data: preProcessedPredictionData, testFraction: 0.99);

            IDataView predictionSet = predictionTest.TestSet;
            
            TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.4);
            TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

            IDataView trainSet = trainSplit.TrainSet;
            IDataView validationSet = validationTestSplit.TrainSet;
            IDataView testSet = validationTestSplit.TestSet;

            var classifierOptions = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                ValidationSet = validationSet,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false,
                ReuseTrainSetBottleneckCachedValues = true,
                ReuseValidationSetBottleneckCachedValues = true
            };

            var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(trainSet);

            Console.WriteLine("=======================================");
            Console.WriteLine("Trying single image");
            RunSingleImage(mlContext, testSet, trainedModel);
            Console.WriteLine("How did that work?");
            Console.WriteLine();
            Console.Write("Input amount of images to try on: ");
            int amount;
            int.TryParse(Console.ReadLine(), out amount);
            Console.WriteLine("=======================================");
            Console.WriteLine($"Trying {amount} random images");
            RunMultipleImages(mlContext, testSet, trainedModel, amount);
            Console.WriteLine("=======================================");
            Console.WriteLine("Trying to classify user images.....");
            RunUserImages(mlContext, predictionSet, trainedModel);
            Console.WriteLine();
            Console.WriteLine("HEUREKA! IT'S DONE! o_O");
        }

        private static void RunUserImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            IDataView predictionData = trainedModel.Transform(data);

            IEnumerable<ModelOutput> predictions = mlContext.Data
                .CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true);
            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction, false);
            }
        }

        private static void RunSingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            PredictionEngine<ModelInput, ModelOutput> predictionEngine =
                mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();

            ModelOutput prediction = predictionEngine.Predict(image);

            OutputPrediction(prediction, true);
        }

        private static void RunMultipleImages(MLContext mlContext, IDataView data, ITransformer trainedModel,
            int amount)
        {
            IDataView predictionData = trainedModel.Transform(data);

            IEnumerable<ModelOutput> predictions = mlContext.Data
                .CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(amount);

            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction, true);
            }
        }

        private static void OutputPrediction(ModelOutput prediction, bool testData)
        {
            string imageName = Path.GetFileName(prediction.ImagePath);
            if (testData) Console.WriteLine($"Image: {imageName} \t| Actual Value: {prediction.Label} \t| Predicted Value: {prediction.PredictedLabel}");
            if (!testData) Console.WriteLine($"Image: {imageName} \t| Predicted Value: {prediction.PredictedLabel}");
        }

        private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);

                if (useFolderNameAsLabel)
                {
                    label = Directory.GetParent(file).Name;
                }
                else
                {
                    for (var i = 0; i < label.Length; i++)
                    {
                        if (!char.IsLetter(label[i]))
                        {
                            label = label.Substring(0, i);
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }
    }

    class ImageData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    class ModelInput
    {
        public byte[] Image { get; set; }
        public UInt32 LabelAsKey { get; set; }
        public string ImagePath { get; set; }
        public string Label { get; set; }
    }

    class ModelOutput
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public string PredictedLabel { get; set; }
    }
}