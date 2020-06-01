using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLNetUsedCarsModel
{
    class AutoSale
    {
        [LoadColumn(0)]
        public string dateCreawled;

        [LoadColumn(1)]
        public string name;

        [LoadColumn(2)]
        public float price;

        [LoadColumn(3)]
        public string vehicleType;

        [LoadColumn(4)]
        public float yearOfRegistration;

        [LoadColumn(5)]
        public string gearbox;

        [LoadColumn(6)]
        public float powerPS;

        [LoadColumn(7)]
        public string model;

        [LoadColumn(8)]
        public float kilometer;

        [LoadColumn(10)]
        public string fuelType;

        [LoadColumn(11)]
        public string brand;

        [LoadColumn(12)]
        public string notRepairedDamage;
    }
}
