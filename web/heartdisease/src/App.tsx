import { Field, Form, Formik } from "formik";
import styled from "styled-components";
import axios from "axios";
import { useEffect, useState } from "react";

const ALL = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    padding: 20px;
    max-width: 1600px;
    form {
        background-color: #fafafa;
        border-radius: 5px;
        broder: 1px solid #eaeaea;
        padding: 20px;
        label,
        input,
        button {
            font-size: 16px;
            padding: 0.5rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
            width: 100%;
            background-color: #f1f4f9;
            text-decoration: none;
            border: none;
        }
        button {
            background-color: #657ef8;
            color: #fff;
            :hover {
                color: #fff;
                font-weight: bold;
            }
        }
        div {
            display: flex;
            label {
                display: flex;
                align-items: baseline;
                input {
                    width: 10%;
                }
            }
        }
    }
`;

const get = function (callback: any) {
    axios({
        url: `http://localhost:3001/api/get`,
        method: "get",
    }).then((result) => {
        callback(result.data);
    });
};

function App() {
    const [logicdata, setLogicdata] = useState<any>([]);
    useEffect(() => {
        get((data: any) => {
            setLogicdata(data[0]);
           
        });
    }, []);
    console.log(logicdata);
    return (
        <ALL>
            <h1>Heart Disease prediction</h1>
            <Formik
                initialValues={{
                    BMI: 0,
                    hutthuoc: 0,
                    uongruou: 0,
                    dotquy: 0,
                    PhysicalHealth: 0,
                    MentalHealth: 0,
                    DiffWalking: 0,
                    gioitinh: 0,
                    tuoi: 0,
                    sactoc: 0,
                    tieuduong: 0,
                    PhysicalActivity: 0,
                    GenHealth: 0,
                    thoigianngu: 0,
                    hen: 0,
                    than: 0,
                    SkinCancer: 0,
                }}
                onSubmit={(values) => {
                    let BMI_convert;
                    if (values.BMI > 40) BMI_convert = 8 / 8;
                    else if (values.BMI >= 35 && values.BMI < 40) BMI_convert = 7 / 8;
                    else if (values.BMI >= 30 && values.BMI < 35) BMI_convert = 6 / 8;
                    else if (values.BMI >= 25 && values.BMI < 30) BMI_convert = 5 / 8;
                    else if (values.BMI >= 18.5 && values.BMI < 25) BMI_convert = 4 / 8;
                    else if (values.BMI >= 17 && values.BMI < 18.5) BMI_convert = 3 / 8;
                    else if (values.BMI >= 16 && values.BMI < 17) BMI_convert = 2 / 8;
                    else BMI_convert = 1 / 8;
                    console.log(BMI_convert);

                    let tuoi_convert;
                    if (values.tuoi > 60) tuoi_convert = 3 / 3;
                    else if (values.tuoi >= 40 && values.tuoi < 60) tuoi_convert = 2 / 3;
                    else tuoi_convert = 1 / 3;
                    console.log(tuoi_convert);

                    let thoigianngu_convert = 0;
                    if (values.tuoi < 65) {
                        if (values.thoigianngu >= 7 && values.thoigianngu <= 9) thoigianngu_convert = 0.5;
                        else if (values.thoigianngu > 9) thoigianngu_convert = 1;
                    } else if (values.tuoi >= 65) {
                        if (values.thoigianngu >= 7 && values.thoigianngu <= 8) thoigianngu_convert = 0.5;
                        else if (values.thoigianngu > 8) thoigianngu_convert = 1;
                    } else thoigianngu_convert = 0;
                    console.log(thoigianngu_convert);

                    let res =
                        parseFloat(logicdata.a) +
                        BMI_convert * parseFloat(logicdata.BMI) +
                        values.hutthuoc * parseFloat(logicdata.Smoking) +
                        values.uongruou * parseFloat(logicdata.AlcoholDrinking) +
                        values.dotquy * parseFloat(logicdata.Stroke) +
                        (values.PhysicalHealth / 100) * parseFloat(logicdata.PhysicalHealth) +
                        (values.MentalHealth / 100) * parseFloat(logicdata.MentalHealth) +
                        values.DiffWalking * parseFloat(logicdata.DiffWalking) +
                        values.gioitinh * parseFloat(logicdata.Sex) +
                        tuoi_convert * parseFloat(logicdata.AgeCategory) +
                        (values.sactoc / 5) * parseFloat(logicdata.Race) +
                        (values.tieuduong / 3) * parseFloat(logicdata.Diabetic) +
                        values.PhysicalActivity * parseFloat(logicdata.PhysicalActivity) +
                        (values.GenHealth / 4) * parseFloat(logicdata.GenHealth) +
                        thoigianngu_convert * parseFloat(logicdata.SleepTime) +
                        values.hen * parseFloat(logicdata.Asthma) +
                        values.than * parseFloat(logicdata.KidneyDisease) +
                        values.SkinCancer * parseFloat(logicdata.SkinCancer);
                    let final= 1/(1+Math.exp(-res));
                    // d??ng 140 ch??a ch???c ch???n
                    if(final>0.5) {
                        alert('T??? l??? m???c b???nh tim c???a b???n l?? : ' + final*100 +'% nguy c?? m???c b???nh c???a b???n cao v???i t??? l??? ch??nh x??c l?? : ' + logicdata.per +'%')
                    }else{
                        alert('T??? l??? m???c b???nh tim c???a b???n l?? : ' + final*100 +'% nguy c?? m???c b???nh c???a b???n th???p v???i t??? l??? ch??nh x??c l?? : ' + logicdata.per +'%')
                    }
                    
                }}
            >
                {() => (
                    <Form>
                        <label htmlFor="bmi">BMI</label>
                        <Field name="BMI" type="number" id="bmi" />
                        <p>Ch??? s??? kh???i c?? th???</p>

                        <label htmlFor="hutthuoc">H??t thu???c</label>
                        <div role="group" aria-labelledby="my-radio-group" id="hutthuoc">
                            <label>
                                <Field type="radio" name="hutthuoc" value="1" />
                                <p>C??</p>
                            </label>
                            <label>
                                <Field type="radio" name="hutthuoc" value="0" />
                                <p>Kh??ng</p>
                            </label>
                        </div>
                        <p>B???n c?? h??t thu???c kh??ng ?</p>

                        <label htmlFor="uongruou">U???ng r?????u</label>
                        <div role="group" aria-labelledby="my-radio-group" id="uongruou">
                            <label>
                                <Field type="radio" name="uongruou" value="1" />
                                <p>C??</p>
                            </label>
                            <label>
                                <Field type="radio" name="uongruou" value="0" />
                                <p>Kh??ng</p>
                            </label>
                        </div>
                        <p>B???n c?? u???ng r?????u kh??ng ?</p>

                        <label htmlFor="dotquy">?????t qu???</label>
                        <div role="group" aria-labelledby="my-radio-group" id="dotquy">
                            <label>
                                <Field type="radio" name="dotquy" value="1" />
                                <p>C??</p>
                            </label>
                            <label>
                                <Field type="radio" name="dotquy" value="0" />
                                <p>Kh??ng</p>
                            </label>
                        </div>
                        <p>B???n ???? bao gi??? b??? ?????t qu??? ch??a ?</p>

                        <label htmlFor="PhysicalHealth">S???c kho??? th??? ch???t</label>
                        <Field name="PhysicalHealth" type="number" id="PhysicalHealth" />
                        <p>
                            B??y gi???, h??y ngh?? v??? s???c kh???e th??? ch???t c???a b???n, bao g???m c??? b???nh t???t v?? th????ng t??ch, trong
                            bao nhi??u ng??y trong su???t 30 ng??y qua
                        </p>

                        <label htmlFor="MentalHealth">S???c kh???e tinh th???n</label>
                        <Field name="MentalHealth" type="number" id="MentalHealth" />
                        <p>
                            Suy ngh?? v??? s???c kh???e tinh th???n c???a b???n, trong 30 ng??y qua s???c kh???e tinh th???n c???a b???n kh??ng
                            t???t l?? bao nhi??u ng??y? (0-30 ng??y)
                        </p>

                        <label htmlFor="DiffWalking">??i b??? g???p kh?? kh??n</label>
                        <div role="group" aria-labelledby="my-radio-group" id="DiffWalking">
                            <label>
                                <Field type="radio" name="DiffWalking" value="1" />
                                <p>C??</p>
                            </label>
                            <label>
                                <Field type="radio" name="DiffWalking" value="0" />
                                <p>Kh??ng</p>
                            </label>
                        </div>
                        <p>B???n c?? g???p kh?? kh??n nghi??m tr???ng khi ??i b??? ho???c leo c???u thang kh??ng?</p>

                        <label htmlFor="gioitinh">Gi???i t??nh</label>
                        <div role="group" aria-labelledby="my-radio-group" id="gioitinh">
                            <label>
                                <Field type="radio" name="gioitinh" value="1" />
                                <p>N???</p>
                            </label>
                            <label>
                                <Field type="radio" name="gioitinh" value="0" />
                                <p>Nam</p>
                            </label>
                        </div>

                        <label htmlFor="tuoi">Tu???i</label>
                        <Field name="tuoi" type="number" id="tuoi" />

                        <label htmlFor="sactoc">S???c t???c</label>
                        <div role="group" aria-labelledby="my-radio-group" id="sactoc">
                            <label>
                                <Field type="radio" name="sactoc" value="5" />
                                <p>White</p>
                            </label>
                            <label>
                                <Field type="radio" name="sactoc" value="4" />
                                <p>Black</p>
                            </label>
                            <label>
                                <Field type="radio" name="sactoc" value="3" />
                                <p>Asian</p>
                            </label>
                            <label>
                                <Field type="radio" name="sactoc" value="2" />
                                <p>American Indian/Alaskan Native</p>
                            </label>
                            <label>
                                <Field type="radio" name="sactoc" value="1" />
                                <p>Other</p>
                            </label>
                            <label>
                                <Field type="radio" name="sactoc" value="0" />
                                <p>Hispanic</p>
                            </label>
                        </div>

                        <label htmlFor="tieuduong">Ti???u ???????ng</label>
                        <div role="group" aria-labelledby="my-radio-group" id="tieuduong">
                            <label>
                                <Field type="radio" name="tieuduong" value="3" />
                                <p>C??</p>
                            </label>
                            <label>
                                <Field type="radio" name="tieuduong" value="2" />
                                <p>Yes (during pregnancy)</p>
                            </label>
                            <label>
                                <Field type="radio" name="tieuduong" value="1" />
                                <p>No, borderline diabetes</p>
                            </label>
                            <label>
                                <Field type="radio" name="tieuduong" value="0" />
                                <p>Kh??ng</p>
                            </label>
                        </div>
                        <p>B???n c?? b??? ti???u ???????ng kh??ng ?</p>

                        <label htmlFor="PhysicalActivity">Ho???t ?????ng th??? ch???t</label>
                        <div role="group" aria-labelledby="my-radio-group" id="PhysicalActivity">
                            <label>
                                <Field type="radio" name="PhysicalActivity" value="1" />
                                <p>C??</p>
                            </label>
                            <label>
                                <Field type="radio" name="PhysicalActivity" value="0" />
                                <p>Kh??ng</p>
                            </label>
                        </div>
                        <p>B???n c?? th?????ng xuy??n v???n ?????ng kh??ng ?</p>

                        <label htmlFor="GenHealth">S???c kh???e hi???n t???i</label>
                        <div role="group" aria-labelledby="my-radio-group" id="GenHealth">
                            <label>
                                <Field type="radio" name="GenHealth" value="4" />
                                <p>Excellent</p>
                            </label>
                            <label>
                                <Field type="radio" name="GenHealth" value="3" />
                                <p>Very good</p>
                            </label>
                            <label>
                                <Field type="radio" name="GenHealth" value="2" />
                                <p>Good</p>
                            </label>
                            <label>
                                <Field type="radio" name="GenHealth" value="1" />
                                <p>Fair</p>
                            </label>
                            <label>
                                <Field type="radio" name="GenHealth" value="0" />
                                <p>Poor</p>
                            </label>
                        </div>
                        <p>B???n c?? th??? n??i r???ng nh??n chung s???c kh???e c???a b???n l?? ...</p>

                        <label htmlFor="thoigianngu">Th???i gian ng???</label>
                        <Field name="thoigianngu" type="number" id="thoigianngu" />

                        <label htmlFor="hen">Hen suy???n</label>
                        <div role="group" aria-labelledby="my-radio-group" id="hen">
                            <label>
                                <Field type="radio" name="hen" value="1" />
                                <p>C??</p>
                            </label>
                            <label>
                                <Field type="radio" name="hen" value="0" />
                                <p>Kh??ng</p>
                            </label>
                        </div>
                        <p>B???n c?? b??? hen suy???n kh??ng ?</p>

                        <label htmlFor="than">B???nh th???n</label>
                        <div role="group" aria-labelledby="my-radio-group" id="than">
                            <label>
                                <Field type="radio" name="than" value="1" />
                                <p>C??</p>
                            </label>
                            <label>
                                <Field type="radio" name="than" value="0" />
                                <p>Kh??ng</p>
                            </label>
                        </div>
                        <p>B???n c?? b??? b??nh th???n kh??ng ?</p>

                        <label htmlFor="SkinCancer">Ung th?? da</label>
                        <div role="group" aria-labelledby="my-radio-group" id="SkinCancer">
                            <label>
                                <Field type="radio" name="SkinCancer" value="1" />
                                <p>C??</p>
                            </label>
                            <label>
                                <Field type="radio" name="SkinCancer" value="0" />
                                <p>Kh??ng</p>
                            </label>
                        </div>
                        <p>B???n c?? b??? ung th?? da kh??ng ?</p>

                        <button type="submit">Ki???m tra</button>
                    </Form>
                )}
            </Formik>
        </ALL>
    );
}

export default App;
