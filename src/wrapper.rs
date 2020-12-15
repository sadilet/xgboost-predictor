use pyo3::prelude::*;
use numpy::PyReadonlyArray2;

// use crate::fvec::{FVecArray};
use crate::predictor::Predictor;

use std::time::{Duration, Instant};


#[pyclass]
pub struct PredictorWrapper {
    pub predictor: Predictor,
}

#[pymethods]
impl PredictorWrapper {
    // #[args(ntree_limit = "0", margin = "false")]
    // pub fn predict(&self, data: Vec<f32>, ntree_limit: usize, margin: bool) -> PyResult<Vec<f32>> {
    //     Ok(self
    //         .predictor
    //         .predict(&FVecArray::new(data), margin, ntree_limit))
    // }

    #[args(ntree_limit = "0", margin = "false")]
    pub fn predict_batch(
        &self,
        data: PyReadonlyArray2<f32>,
        ntree_limit: usize,
        margin: bool,
    ) -> PyResult<Vec<Vec<f32>>>  {
        let start = Instant::now();
        let row_length = data.shape()[1] - 1 as usize;
        let preds = self.predictor.predict_many(data.as_slice().unwrap(), row_length, margin, ntree_limit);
        let stop = start.elapsed();
        println!("{:?}", stop);
        Ok(preds)
    }

    // #[args(ntree_limit = "0", margin = "false")]
    // pub fn predict_single(
    //     &self,
    //     data: Vec<f32>,
    //     ntree_limit: usize,
    //     margin: bool,
    // ) -> PyResult<f32> {
    //     Ok(self
    //         .predictor
    //         .predict_single(&FVecArray::new(data), margin, ntree_limit))
    // }

    // #[args(ntree_limit = "0")]
    // pub fn predict_leaf(&self, data: Vec<f32>, ntree_limit: usize) -> PyResult<Vec<usize>> {
    //     Ok(self
    //         .predictor
    //         .predict_leaf(&FVecArray::new(data), ntree_limit))
    // }
}
