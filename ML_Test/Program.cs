using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;

namespace ML_Test
{
    class Program
    {
        private static IDataView predictImageData;
        private static IDataView imageData;
        [SuppressMessage("ReSharper.DPA", "DPA0003: Excessive memory allocations in LOH", MessageId = "type: System.Byte[]")]
        static void Main(string[] args)
        {
            // Step out of bin/debug to root dir
            var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
            var assetsRelativePath = Path.Combine(projectDirectory, "assets"); // Training data
            var imageToPredictPath = Path.Combine(projectDirectory, "predict");// Images to predict

            MLContext mlContext = new MLContext();

            // Loading images for training + assesment/prediction
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
            IEnumerable<ImageData> imagesToPredict = LoadImagesFromDirectory(folder: imageToPredictPath, useFolderNameAsLabel: true);

            imageData = mlContext.Data.LoadFromEnumerable(images);
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

            // Processing training images + prediction images
            IDataView preProcessedData = preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);
            IDataView preProcessedPredictionData = preprocessingPredictionPipeline.Fit(predictImageData).Transform(predictImageData);
            TrainTestData predictionTest = mlContext.Data.TrainTestSplit(data: preProcessedPredictionData, testFraction: 0.99);
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

            var trainingPipeline = mlContext.MulticlassClassification.Trainers
                .ImageClassification(classifierOptions)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            ITransformer trainedModel = trainingPipeline.Fit(trainSet); // Starting the training process!

            Console.WriteLine("=======================================");
            Console.WriteLine("Trying single image");
            // Trying to assess one image from the testSet, comparing it to actual result
            RunSingleImage(mlContext, testSet, trainedModel);
            
            Console.Write("Input amount of images to try on: ");
            if (int.TryParse(Console.ReadLine(), out int amount)) // If TryParse fails, it skips
            {
                Console.WriteLine("=======================================");
                Console.WriteLine($"Trying {amount} random images");
                // Trying to assess n image/s from the testSet, comparing it to actual result
                RunMultipleImages(mlContext, testSet, trainedModel, amount);
            }
            
            Console.WriteLine("=======================================");
            Console.WriteLine("Trying to classify user images....."); 
            RunUserImages(mlContext, predictionSet, trainedModel); // Using the images in the predictionSet and guesses what it is
            
            Console.WriteLine();
            Console.WriteLine("HEUREKA! IT'S DONE! o_O");
        }
        
        /// <summary>
        /// Runs a single image from the training-set to check if the ML works
        /// </summary>
        /// <param name="mlContext">ML Contect</param>
        /// <param name="data">Dataset</param>
        /// <param name="trainedModel">Model to use</param>
        private static void RunSingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
        {
            PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
            
            ModelOutput prediction = predictionEngine.Predict(mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First());

            OutputPrediction(prediction, true);
        }

        /// <summary>
        /// Runs the image/s from the predict folder to guess what it is
        /// </summary>
        /// <param name="mlContext">ML Context</param>
        /// <param name="data">Dataset</param>
        /// <param name="trainedModel">Model to use</param>
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

        /// <summary>
        /// Runs multiple images from the training-set to check the accuracy of the model
        /// </summary>
        /// <param name="mlContext">ML Context</param>
        /// <param name="data">Dataset</param>
        /// <param name="trainedModel">Model to use</param>
        /// <param name="amount">Amount of images to test</param>
        private static void RunMultipleImages(MLContext mlContext, IDataView data, ITransformer trainedModel, int amount)
        {
            var predictionData = trainedModel.Transform(data);

            IEnumerable<ModelOutput> predictions = mlContext.Data
                .CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true)
                .Take(amount);

            foreach (var prediction in predictions)
            {
                OutputPrediction(prediction, true);
            }
        }

        /// <summary>
        /// Outputs the prediction to console log
        /// </summary>
        /// <param name="prediction">Predicted data</param>
        /// <param name="testData">Is using test-data?</param>
        private static void OutputPrediction(ModelOutput prediction, bool testData)
        {
            var imageName = Path.GetFileName(prediction.ImagePath);
            switch (testData)
            {
                case true:
                    Console.WriteLine($"Image: {imageName} \t| Actual Value: {prediction.Label} \t| Predicted Value: {prediction.PredictedLabel}");
                    break;
                case false:
                    Console.WriteLine($"Image: {imageName} \t| Predicted Value: {prediction.PredictedLabel} \t| Folder: {prediction.Label}");
                    break;
            }
        }

        /// <summary>
        /// Loads images from directory/ies to an enum
        /// </summary>
        /// <param name="folder">Path of the folder to import</param>
        /// <param name="useFolderNameAsLabel">Using the folder-name as label on the images</param>
        /// <returns></returns>
        private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png") &&
                    (Path.GetExtension(file) != ".jpeg"))
                {
                    continue;
                }

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