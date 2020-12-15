#![recursion_limit = "1024"]

#[macro_use]
extern crate error_chain;

pub mod errors {
    error_chain! {
        foreign_links {
            Io(::std::io::Error);
            Utf8Error(::std::string::FromUtf8Error);
        }
        // Define additional `ErrorKind` variants.  Define custom responses with the
        // `description` and `display` calls.
        errors {
            UnsupportedModelType(t: String) {
                description("Unsupported model type")
                display("Unsupported model type: '{}'", t)
            }
            UnsupportedObjFunctionType(t: String) {
                description("Unsupported object function type")
                display("Unsupported object function type: '{}'", t)
            }
        }
    }
}

mod functions;
// pub mod fvec;
mod gbm;
pub mod model_reader;
pub mod predictor;
mod wrapper;

use pyo3::prelude::*;
use pyo3::{create_exception, exceptions, wrap_pyfunction, PyErr};

use std::{fs, io};

#[pyfunction]
fn load_model(py: Python, model_path: &str) -> PyResult<wrapper::PredictorWrapper> {
    let model_file = fs::File::open(model_path);
    let mut model_file = match model_file {
        Ok(file) => file,
        Err(error) => match error.kind() {
            io::ErrorKind::NotFound => {
                return Err(PyErr::new::<exceptions::PyFileNotFoundError, _>(format!(
                    "File not found: {}",
                    model_path
                )))
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyOSError, _>(format!(
                    "Unexpected: {}",
                    error
                )))
            }
        },
    };
    let predictor = predictor::Predictor::read_from::<fs::File>(&mut model_file);
    match predictor {
        Err(error) => Err(PyErr::new::<exceptions::PyOSError, _>(format!(
            "Unexpected: {}",
            error
        ))),
        Ok(_) => Ok(wrapper::PredictorWrapper {
            predictor: predictor.unwrap(),
        }),
    }
}

#[pymodule]
fn xgboost_predictor(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_model, m)?)?;

    Ok(())
}
