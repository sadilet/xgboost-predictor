use crate::errors::*;

pub enum FunctionType {
    RankPairwise,
    BinaryLogistic,
    BinaryLogitraw,
    MultiSoftmax,
    MultiSoftprob,
    RegLinear,
}

/// interface of objective function
pub struct ObjFunction {
    pub vector: fn(&Vec<f32>) -> Vec<f32>,
    pub scalar: fn(f32) -> f32,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1f32 / (1f32 + (-x).exp())
}

fn dump_vec(preds: &Vec<f32>) -> Vec<f32> {
    return preds.clone();
}

fn dump(pred: f32) -> f32 {
    return pred;
}

/// Logistic regression.
fn logistic_vec(preds: &Vec<f32>) -> Vec<f32> {
    return preds.into_iter().map(|x| sigmoid(*x)).collect();
}

/// Multiclass classification.
fn multiclass_vec(preds: &Vec<f32>) -> Vec<f32> {
    match preds.get(0) {
        Option::Some(init) => {
            let (max_index, _max) =
                preds
                    .iter()
                    .enumerate()
                    .fold(
                        (0, init),
                        |(i1, v1), (i2, v2)| if v1 >= v2 { (i1, v1) } else { (i2, v2) },
                    );
            return vec![max_index as f32; 1];
        }
        // empty vector
        Option::None => preds.clone(),
    }
}

fn unimplemented(_pred: f32) -> f32 {
    std::unimplemented!();
}

///  Multiclass classification (predicted probability).
fn multiclass_pred_prob_vec(preds: &Vec<f32>) -> Vec<f32> {
    match preds.get(0) {
        Option::Some(init) => {
            let max = preds.iter().fold(*init, |a, b| b.max(a));
            let sum: f32 = preds.iter().map(|x| (x - max).exp()).sum();
            return preds.iter().map(|x| (x - max).exp() / sum).collect();
        }
        // empty vector
        Option::None => preds.clone(),
    }
}

pub fn get_classify_function(tp: FunctionType) -> ObjFunction {
    match tp {
        FunctionType::RankPairwise | FunctionType::BinaryLogitraw | FunctionType::RegLinear => {
            ObjFunction {
                vector: dump_vec,
                scalar: dump,
            }
        }
        FunctionType::BinaryLogistic => ObjFunction {
            vector: logistic_vec,
            scalar: sigmoid,
        },
        FunctionType::MultiSoftmax => ObjFunction {
            vector: multiclass_vec,
            scalar: unimplemented,
        },
        FunctionType::MultiSoftprob => ObjFunction {
            vector: multiclass_pred_prob_vec,
            scalar: unimplemented,
        },
    }
}

pub fn get_classify_func_type(obj_name: Vec<u8>) -> Result<FunctionType> {
    return match obj_name.as_slice() {
        b"rank:pairwise" => Ok(FunctionType::RankPairwise),
        b"binary:logistic" => Ok(FunctionType::BinaryLogistic),
        b"binary:logitraw" => Ok(FunctionType::BinaryLogitraw),
        b"multi:softmax" => Ok(FunctionType::MultiSoftmax),
        b"multi:softprob" => Ok(FunctionType::MultiSoftprob),
        b"reg:linear" => Ok(FunctionType::RegLinear),
        _ => Err(Error::from_kind(ErrorKind::UnsupportedObjFunctionType(
            String::from_utf8(obj_name)?,
        ))),
    };
}

#[cfg(test)]
mod tests {
    use crate::functions::get_classify_function;
    use crate::functions::FunctionType::BinaryLogistic;

    #[test]
    fn test_get_classify_function() {
        let func = get_classify_function(BinaryLogistic);
        (func.vector)(&vec![1.0f32, 4.6f32]);
    }
}
