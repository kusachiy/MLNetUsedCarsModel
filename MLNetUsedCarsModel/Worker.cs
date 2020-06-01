using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetUsedCarsModel
{
    public static class Worker
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "autos_train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "autos_eval.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static readonly AutoSale _test = new AutoSale
        {
            dateCreawled = "2016-03-19 18:57:12",
            name = "Volkswagen_Multivan_T4_TDI_7DC_UY2",
            vehicleType = "bus",
            yearOfRegistration = 1999f,
            gearbox = "manuell",
            powerPS = 149,
            model = "transporter",
            kilometer = 150000f,
            fuelType = "diesel",
            brand = "volkswagen",
            notRepairedDamage = "nein",
            price = 0
        };
        
        static readonly AutoSale _test2 = new AutoSale()
        {
            dateCreawled = "2016-03-20 19:41:08",
            name = "VW_Golf_Kombi_1_9l_TDI",
            vehicleType = "kombi",
            yearOfRegistration = 2000f,
            gearbox = "manuell",
            powerPS = 100,
            model = "golf",
            kilometer = 150000f,
            fuelType = "diesel",
            brand = "volkswagen",
            notRepairedDamage = "ja",
            price = 0
        };
        static readonly AutoSale _test3 = new AutoSale()
        {
            dateCreawled = "2016-03-07 19:39:19",
            name = "BMW_M135i_vollausgestattet_NP_52",
            vehicleType = "720____Euro",
            yearOfRegistration = 2013,
            gearbox = "manuell",
            powerPS = 320,
            model = "limousine",
            kilometer = 50000,
            fuelType = "benzin",
            brand = "bmw",
            notRepairedDamage = "nein",
            price = 0
        };

        private static Action<string> writeline;
        private static MLContext _mlContext;
        private static ITransformer _model;

        public static void Work(Action<string> action)
        {
            writeline = action;           
        }

        public static void Train()
        {
            _mlContext = new MLContext(seed: 23752712);
            _model = Train(_mlContext, _trainDataPath);
            writeline("Training complete...");
        }
        public static void Evaluate()
        {
            Evaluate(_mlContext, _model);
        }
        public static void Test1()
        {
            TestSinglePrediction(_test, 6250);
        }
        public static void Test2()
        {
            TestSinglePrediction(_test2, 12700);
        }
        public static void Test3()
        {
            TestSinglePrediction(_test3, 23990);
        }


        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // The IDataView object holds the training dataset 
            IDataView dataView = mlContext.Data.LoadFromTextFile<AutoSale>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "price")
                 .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "nameEncoded", inputColumnName: "name"))
               .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "vehicleTypeEncoded", inputColumnName: "vehicleType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "gearboxEncoded", inputColumnName: "gearbox"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "modelEncoded", inputColumnName: "model"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "fuelTypeEncoded", inputColumnName: "fuelType"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "brandEncoded", inputColumnName: "brand"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "notRepairedDamageEncoded", inputColumnName: "notRepairedDamage"))

                .Append(mlContext.Transforms.Concatenate("Features", "powerPS", "kilometer", "nameEncoded",
                "vehicleTypeEncoded",
                "gearboxEncoded",
                "modelEncoded", "yearOfRegistration", "fuelTypeEncoded", "brandEncoded", "notRepairedDamageEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

            //Create the model
            var model = pipeline.Fit(dataView);

            //Return the trained model
            return model;
        }
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<AutoSale>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            writeline("");
            writeline($"*************************************************");
            writeline($"*       Model quality metrics output         ");
            writeline($"*------------------------------------------------");

            writeline($"*       R-Squared Score:      {metrics.RSquared:0.###}");

            writeline($"*       Root-Mean-Squared Error:      {metrics.RootMeanSquaredError:#.###}");
        }
        private static void TestSinglePrediction(AutoSale autoSale, float? realPrice)
        {
            var predictionFunction = _mlContext.Model.CreatePredictionEngine<AutoSale, AutoSalePrediction>(_model);
            var taxiTripSample = autoSale;
            var prediction = predictionFunction.Predict(taxiTripSample);
            writeline($"**********************************************************************");
            writeline($"Predicted cost is: {prediction.price:0.####}, while actual price: {realPrice}");
            writeline($"**********************************************************************");
        }
    }
}
