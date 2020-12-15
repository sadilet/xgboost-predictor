use std::collections::HashMap;
use std::{f32};


/// Interface of feature vector
// pub trait FVec {
//     /// get value for index
//     fn fvalue(&self, index: usize, f_position: usize) -> Option<f32>;
//     fn rl(&self) -> usize;
// }

// pub type FVecMap<T: ToFloat> = HashMap<usize, T>;

// Feature vector based on vec
// pub struct FVecArray<T: ToFloat> {
//     values: Vec<T>,
//     row_length: usize,
//     treats_zero_as_none: bool,
// }

// impl<T: ToFloat> FVec for FVecMap<T> {
//     fn fvalue(&self, index: usize) -> Option<f32> {
//         return Some(self.get(&index)?.to_double());
//     }
// }

// impl<T: ToFloat> FVec for FVecArray<T> {
//     fn fvalue(&self, index: usize, f_position: usize) -> Option<f32> {
//         if self.values.len() <= index {
//             return None;
//         } else {
//             let result = self.values[(f_position + (self.row_length * f_position)) + index].to_double();
//             if self.treats_zero_as_none && result == 0f32 {
//                 return None;
//             } else {
//                 return Some(result);
//             }
//         }
//     }

//     fn rl(&self) -> usize {
//         self.values.len()
//     }
// }


// impl<T: ToFloat> FVecArray<T> {
//     pub fn new(values: Vec<T>) -> Self {
//         FVecArray {
//             values: values,
//             treats_zero_as_none: true,
//             row_length: 125,
//         }
//     }
// }
