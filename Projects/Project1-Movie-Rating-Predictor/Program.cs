using Microsoft.ML;
using Project1_Movie_Rating_Predictor;

var mlContext = new MLContext();

var data = mlContext.Data.LoadFromTextFile<MovieData>(
    "movies_10000.csv",
    hasHeader: true,
    separatorChar: ','
);

// Prepare pipeline
var pipeline =
    mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "GenreEncoded",
        inputColumnName: "Genre")
    .Append(mlContext.Transforms.Concatenate("Features", "Budget", "ActorPopularity", "GenreEncoded"))
    .Append(mlContext.Regression.Trainers.FastTree(
        labelColumnName: "Rating",
        featureColumnName: "Features"));

// Train model
var model = pipeline.Fit(data);

// Save model
mlContext.Model.Save(model, data.Schema, "MovieRatingModel.zip");

Console.WriteLine("Model trained and saved!");

var predictor = mlContext.Model.CreatePredictionEngine<MovieData, MoviePrediction>(model);

var input = new MovieData
{
    Budget = 120000000,
    ActorPopularity = 0.9f,
    Genre = "Action"
};

var result = predictor.Predict(input);

Console.WriteLine($"Predicted Rating: {result.Score}");

Console.ReadLine();