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
mod gbm;
pub mod model_reader;
pub mod predictor;
mod wrapper;

use std::{fs, io};

use pyo3::prelude::*;
use pyo3::{exceptions, wrap_pyfunction, PyErr};

#[pyfunction]
fn load_model(model_path: &str) -> PyResult<wrapper::PredictorWrapper> {
    let mut model_file = match fs::File::open(model_path) {
        Ok(file) => file,
        Err(error) => match error.kind() {
            io::ErrorKind::NotFound => {
                return Err(PyErr::new::<exceptions::PyFileNotFoundError, _>(format!(
                    "File not found: {}.",
                    model_path
                )))
            }
            _ => {
                return Err(PyErr::new::<exceptions::PyOSError, _>(format!("Unexpected error, when open file: {}.", error)))
            }
        },
    };
    match predictor::Predictor::read_from::<fs::File>(&mut model_file) {
        Ok(predictor) => Ok(wrapper::PredictorWrapper {predictor: predictor}),
        Err(error) => match error.kind() {
            errors::ErrorKind::UnsupportedModelType(message) | errors::ErrorKind::UnsupportedObjFunctionType(message) => {
                Err(PyErr::new::<exceptions::PyValueError, _>(message.clone()))
            }
            _ => {
                Err(PyErr::new::<exceptions::PyValueError, _>(format!("Unexpected error, when initializing model: {}.", error)))
            }
        },
    }
}

#[pymodule]
fn xgboost_predictor(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_model, m)?)?;

    Ok(())
}
