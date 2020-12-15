use crate::errors::*;
use byteorder::{ReadBytesExt, BE, LE};
use std::io::Read;

pub trait ModelReader: ReadBytesExt {
    #[inline]
    fn read_i32_le(&mut self) -> Result<i32> {
        return self
            .read_i32::<LE>()
            .chain_err(|| "failed to read i32 from model");
    }

    #[inline]
    fn read_i64_le(&mut self) -> Result<i64> {
        return self
            .read_i64::<LE>()
            .chain_err(|| "failed to read i64 from model");
    }

    #[inline]
    fn read_f32_le(&mut self) -> Result<f32> {
        return self
            .read_f32::<LE>()
            .chain_err(|| "failed to read f32 from model");
    }

    #[inline]
    fn read_byte_as_i32(&mut self) -> Result<i32> {
        let byte = self.read_u8()?;
        return Ok(byte as i32);
    }

    fn read_to_i32_buffer(&mut self, buffer: &mut [i32]) -> Result<()> {
        for b in buffer.iter_mut() {
            match self.read_i32_le() {
                Ok(val) => *b = val,
                Err(e) => return Err(e),
            }
        }
        return Ok(());
    }

    fn read_to_f32_buffer(&mut self, buffer: &mut [f32]) -> Result<()> {
        for b in buffer.iter_mut() {
            match self.read_f32_le() {
                Ok(val) => *b = val,
                Err(e) => return Err(e),
            }
        }
        return Ok(());
    }
    fn read_to_f64_buffer_be(&mut self, buffer: &mut [f64]) -> Result<()> {
        for b in buffer.iter_mut() {
            *b = self.read_f64::<BE>()?;
        }
        return Ok(());
    }

    fn read_int_vec(&mut self, num_values: usize) -> Result<Vec<i32>> {
        return (0..num_values).map(|_| self.read_i32_le()).collect();
    }

    fn read_float_vec(&mut self, num_values: usize) -> Result<Vec<f32>> {
        return (0..num_values).map(|_| self.read_f32_le()).collect();
    }

    fn skip(&mut self, num_bytes: usize) -> Result<()> {
        let mut vec: Vec<u8> = vec![0u8; num_bytes];
        return self
            .read_exact(vec.as_mut_slice())
            .chain_err(|| "failed to read u8 slice from model");
    }

    fn read_u8_vec(&mut self, size: usize) -> Result<Vec<u8>> {
        let mut vec: Vec<u8> = vec![0u8; size];
        self.read_exact(vec.as_mut_slice())?;
        return Ok(vec);
    }

    fn read_u8_vec_len(&mut self) -> Result<Vec<u8>> {
        let len = self.read_i64_le()? as usize;
        return self.read_u8_vec(len);
    }
}

impl<T: Read> ModelReader for T {}
