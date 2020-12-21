use ndarray::{ArrayView1, ArrayView2};

use crate::errors::*;
use crate::gbm::gblinear::GBLinear;
use crate::gbm::gbtree::GBTree;
use crate::model_reader::ModelReader;

/// Interface of gradient boosting model
pub trait GradBooster {
    /// Generates predictions for given feature vector
    // fn predict(&self, feat: ArrayView1<'_, f32>, ntree_limit: usize) -> Result<Vec<f32>>;
    // /// Generates a prediction for given feature vector
    // fn predict_single(&self, feat: ArrayView1<'_, f32>, ntree_limit: usize) -> Result<f32>;
    // /// Predicts the leaf index of each tree. This is only valid in gbtree predictor
    // fn predict_leaf(&self, feat: ArrayView1<'_, f32>, ntree_limit: usize) -> Result<Vec<usize>>;
    // /// Generates predictions for given vectors of features
    fn predict_many(&self, feats: ArrayView2<'_, f32>, base_score: f32, ntree_limit: usize)
        -> Result<Vec<Vec<f32>>>;
}

pub fn load_grad_booster<T: ModelReader>(
    reader: &mut T,
    name_gbm: Vec<u8>,
    with_pbuffer: bool,
) -> Result<Box<dyn GradBooster + Send>> {
    match name_gbm.as_slice() {
        b"gbtree" => Ok(Box::new(GBTree::read_from(with_pbuffer, reader, false)?)),
        b"gblinear" => Ok(Box::new(GBLinear::read_from(with_pbuffer, reader)?)),
        b"dart" => Ok(Box::new(GBTree::read_from(with_pbuffer, reader, true)?)),
        _ => Err(Error::from_kind(ErrorKind::UnsupportedModelType(
            String::from_utf8(name_gbm)?,
        ))),
    }
}
