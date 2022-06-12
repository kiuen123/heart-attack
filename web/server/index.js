var express = require("express");
var database = require("./db.js");
var app = express();

var currentdate = new Date();
var datetime =
    currentdate.getDate() +
    "/" +
    (currentdate.getMonth() + 1) +
    "/" +
    currentdate.getFullYear() +
    "-" +
    currentdate.getSeconds() +
    ":" +
    currentdate.getMinutes() +
    ":" +
    currentdate.getHours();

app.get("/api/get", function (req,res){
    res.header("Access-Control-Allow-Origin", "*");
    database.get(function(data){
        console.log(data);
            res.send(data);
    })
})

app.get("/api/add", function (req, res) {
    res.header("Access-Control-Allow-Origin", "*");
    a = req.query.a;
    BMI = req.query.BMI;
    Smoking = req.query.Smoking;
    AlcoholDrinking = req.query.AlcoholDrinking;
    Stroke = req.query.Stroke;
    PhysicalHealth = req.query.PhysicalHealth;
    MentalHealth = req.query.MentalHealth;
    DiffWalking = req.query.DiffWalking;
    Sex = req.query.Sex;
    AgeCategory = req.query.AgeCategory;
    Race = req.query.Race;
    Diabetic = req.query.Diabetic;
    PhysicalActivity = req.query.PhysicalActivity;
    GenHealth = req.query.GenHealth;
    SleepTime = req.query.SleepTime;
    Asthma = req.query.Asthma;
    KidneyDisease = req.query.KidneyDisease;
    SkinCancer = req.query.SkinCancer;
    per = req.query.per;
    database.add(
        a,
        BMI,
        Smoking,
        AlcoholDrinking,
        Stroke,
        PhysicalHealth,
        MentalHealth,
        DiffWalking,
        Sex,
        AgeCategory,
        Race,
        Diabetic,
        PhysicalActivity,
        GenHealth,
        SleepTime,
        Asthma,
        KidneyDisease,
        SkinCancer,
        per,
        datetime,
        function (data) {
            console.log(data);
            res.send(data);
        }
    );
});

app.listen(3001);
