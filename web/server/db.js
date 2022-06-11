var mysql = require("mysql");

var connection = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "",
    database: "heartdisease",
});

var connect = function () {
    connection.connect(function (err) {
        if (!err) {
            console.log("db connect ok");
        } else {
            console.log("db connect fail");
        }
    });
};

var close = function () {
    connection.end(function (err) {
        if (!err) {
            console.log("db close");
        } else {
            console.log("db close fail");
        }
    });
};

exports.get = function (callback) {
    connect();
    connection.query("SELECT * FROM logic ORDER BY per DESC LIMIT 1", function (err, res, fields) {
        if (!err) {
            callback(res);
        } else {
            console.log(err);
        }
    });
};

exports.add = function (
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
    date,
    callback
) {
    connect();
    connection.query(
        `INSERT INTO logic(a, BMI, Smoking, AlcoholDrinking, Stroke, PhysicalHealth, MentalHealth, DiffWalking, Sex, AgeCategory, Race, Diabetic, PhysicalActivity, GenHealth, SleepTime, Asthma, KidneyDisease, SkinCancer, per, updateon) 
        VALUES (${a},${BMI},${Smoking},${AlcoholDrinking},${Stroke} ,${PhysicalHealth} ,${MentalHealth} ,${DiffWalking} ,${Sex} ,${AgeCategory} ,${Race} ,${Diabetic} ,${PhysicalActivity} ,${GenHealth} ,${SleepTime} ,${Asthma} ,${KidneyDisease} ,${SkinCancer} ,${per} ,'${date}')`,
        function (err, res, fields) {
            if (!err) {
                callback(res);
            } else {
                console.log(err);
            }
        }
    );
};
