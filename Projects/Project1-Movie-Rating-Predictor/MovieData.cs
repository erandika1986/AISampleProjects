using Microsoft.ML.Data;

namespace Project1_Movie_Rating_Predictor
{
    public class MovieData
    {
        [LoadColumn(0)]
        public float Budget { get; set; }

        [LoadColumn(1)]
        public float ActorPopularity { get; set; }

        [LoadColumn(2)]
        public string Genre { get; set; }

        [LoadColumn(3)]
        public float Rating { get; set; }
    }
}
