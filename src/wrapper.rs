use ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{exceptions, PyErr};

use crate::errors::*;
use crate::predictor::Predictor;

use std::time::{Duration, Instant};

fn check_input_1d(predictor: &Predictor, data: &ArrayView1<'_, f32>) -> PyResult<()> {
    let model_num_features = predictor.model_num_feature();
    let data_num_features = data.shape()[0];
    if model_num_features != data_num_features {
        return Err(PyErr::new::<exceptions::PyValueError, _>(format!(
            "Num of features is not equal to model's features count: {} != {}.",
            data_num_features, model_num_features,
        )));
    }
    Ok(())
}

fn check_input_2d(predictor: &Predictor, data: &ArrayView2<'_, f32>) -> PyResult<()> {
    let model_num_features = predictor.model_num_feature();
    let data_num_features = data.shape()[1];
    if model_num_features != data_num_features {
        return Err(PyErr::new::<exceptions::PyValueError, _>(format!(
            "Num of features is not equal to model's features count: {} != {}.",
            data_num_features, model_num_features,
        )));
    }
    Ok(())
}

#[pyclass]
pub struct PredictorWrapper {
    pub predictor: Predictor,
}

#[pymethods]
impl PredictorWrapper {
    #[args(ntree_limit = "0")]
    pub fn predict_leaf<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f32>,
        ntree_limit: usize,
    ) -> PyResult<&'py PyArray1<usize>> {
        let data_array = data.as_array();
        check_input_1d(&self.predictor, &data_array)?;
        match self.predictor.predict_leaf(data_array, ntree_limit) {
            Ok(preds) => Ok(PyArray1::from_vec(py, preds)),
            Err(error) => match error.kind() {
                _ => Err(PyErr::new::<exceptions::PyValueError, _>("")),
            },
        }
    }

    #[args(ntree_limit = "0", margin = "false")]
    pub fn predict_single(
        &self,
        data: PyReadonlyArray1<f32>,
        ntree_limit: usize,
        margin: bool,
    ) -> PyResult<f32> {
        let data_array = data.as_array();
        check_input_1d(&self.predictor, &data_array)?;
        match self
            .predictor
            .predict_single(data_array, margin, ntree_limit)
        {
            Ok(pred) => Ok(pred),
            Err(error) => match error.kind() {
                _ => Err(PyErr::new::<exceptions::PyValueError, _>("")),
            },
        }
    }

    #[args(ntree_limit = "0", margin = "false")]
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f32>,
        ntree_limit: usize,
        margin: bool,
    ) -> PyResult<&'py PyArray1<f32>> {
        let data_array = data.as_array();
        check_input_1d(&self.predictor, &data_array)?;
        match self.predictor.predict(data_array, margin, ntree_limit) {
            Ok(preds) => Ok(PyArray1::from_vec(py, preds)),
            Err(error) => match error.kind() {
                _ => Err(PyErr::new::<exceptions::PyValueError, _>("")),
            },
        }
    }

    #[args(ntree_limit = "0", margin = "false")]
    pub fn predict_many(
        &self,
        data: PyReadonlyArray2<f32>,
        ntree_limit: usize,
        margin: bool,
    ) -> PyResult<Vec<Vec<f32>>> {
        // let data_array = data.as_array();
        // check_input_2d(&self.predictor, &data_array)?;
        let now = Instant::now();
        let preds = self.predictor.predict_many(data.as_slice().unwrap(), margin, ntree_limit);
        println!("{:?}", now.elapsed());
        Ok(preds.unwrap())
        // match self.predictor.predict_many(data.as_slice().unwrap(), margin, ntree_limit) {
        //     Ok(preds) => Ok(preds),
        //     Err(error) => match error.kind() {
        //         _ => Err(PyErr::new::<exceptions::PyValueError, _>("")),
        //     },
        // }
    }

    // #[args(ntree_limit = "0", margin = "false")]
    // pub fn predict_many(
    //     &self,
    //     data: PyReadonlyArray2<f32>,
    //     ntree_limit: usize,
    //     margin: bool,
    // ) -> PyResult<Vec<Vec<f32>>> {
    //     let data_array = data.as_array();

    //     let now = Instant::now();
    //     check_input_2d(&self.predictor, &data_array)?;
    //     println!("{:?}", now.elapsed());

    //     let now = Instant::now();
    //     let data = self.predictor.predict_many(data_array, margin, ntree_limit);
    //     println!("{:?}", now.elapsed());

    //     match self.predictor.predict_many(data_array, margin, ntree_limit) {
    //         Ok(preds) => Ok(preds),
    //         Err(error) => match error.kind() {
    //             _ => Err(PyErr::new::<exceptions::PyValueError, _>("")),
    //         },
    //     }
    // }
}
