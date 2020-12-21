use std::iter::FromIterator;

use ndarray::{ArrayView1, ArrayView2, Array, Ix1};

use std::time::{Duration, Instant};

use itertools::izip;
use crate::errors::*;
use crate::gbm::grad_booster::GradBooster;
use crate::model_reader::ModelReader;

struct ModelParam {
    /// number of features
    num_feature: usize,
    /// how many output group a single instance can produce
    /// this affects the behavior of number of output we have:
    /// suppose we have n instance and k group, output will be k*n
    num_output_group: usize,
}

impl ModelParam {
    fn read_from<T: ModelReader>(reader: &mut T) -> Result<ModelParam> {
        let (num_feature, num_output_group) = (
            reader.read_i32_le()? as usize,
            reader.read_i32_le()? as usize,
        );
        let mut reserved = [0i32; 32];
        reader.read_to_i32_buffer(&mut reserved)?;
        // read padding
        reader.read_i32_le()?;
        Ok(ModelParam {
            num_feature,
            num_output_group,
        })
    }
}

pub struct GBLinear {
    mparam: ModelParam,
    weights: Vec<f32>,
}

impl GBLinear {
    pub fn read_from<T: ModelReader>(with_pbuffer: bool, reader: &mut T) -> Result<Self> {
        let mparam = ModelParam::read_from(reader)?;
        // read padding
        reader.read_i32_le()?;
        let weights = reader.read_float_vec((mparam.num_feature + 1) * mparam.num_output_group)?;

        Ok(GBLinear { mparam, weights })
    }

    fn bias(&self, gid: usize) -> f32 {
        self.weight(self.mparam.num_feature, gid)
    }

    fn weight(&self, fid: usize, gid: usize) -> f32 {
        self.weights[(fid * self.mparam.num_output_group) + gid]
    }

    fn group_weights(&self, gid: usize) -> Array<f32, Ix1> {
        <Array<f32, Ix1> as FromIterator<f32>>::from_iter((0..self.mparam.num_feature).map(|fid| self.weight(fid, gid)))
    }

    fn pred(&self, feat: ArrayView1<'_, f32>, gid: usize) -> f32 {
        feat.dot(&self.group_weights(gid)) + self.bias(gid as usize)
    }

    fn pred_many(&self, feats: ArrayView2<'_, f32>, base_score: f32,  gid: usize) -> Vec<f32> {
        (feats.dot(&self.group_weights(gid)) + self.bias(gid as usize) + base_score).to_vec()
    }
}

impl GradBooster for GBLinear {
    // fn predict(&self, feat: ArrayView1<'_, f32>, ntree_limit: usize) -> Result<Vec<f32>> {
    //     let mut data: Vec<f32> = vec![];
    //     for gid in 0..self.mparam.num_output_group {
    //         data.push(self.pred(feat, gid)?)
    //     }
    //     Ok(data)
    // }

    // fn predict_single(&self, feat: ArrayView1<'_, f32>, ntree_limit: usize) -> Result<f32> {
    //     if self.mparam.num_output_group != 1 {
    //         return Err(Error::from_kind(ErrorKind::UnsupportedPredictionMethod(
    //             String::from("predict_single"),
    //             format!(
    //                 "Detail: output group must be equal to 1, current value {}",
    //                 self.mparam.num_output_group
    //             ),
    //         )));
    //     }
    //     self.pred(feat, 0)
    // }

    // fn predict_leaf(&self, feat: ArrayView1<'_, f32>, ntree_limit: usize) -> Result<Vec<usize>> {
    //     Err(Error::from_kind(ErrorKind::UnsupportedPredictionMethod(
    //         String::from("predict_leaf"),
    //         format!("Detail: gblinear model does not support predict leaf index",),
    //     )))
    // }

    fn predict_many(
        &self,
        feats: ArrayView2<'_, f32>,
        base_score: f32, 
        ntree_limit: usize,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(izip!((0..self.mparam.num_output_group).map(|gid| self.pred_many(feats, base_score, gid))).collect())
    }
}
