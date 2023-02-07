use anyhow::{anyhow, bail, Result};
use pyo3::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use crate::{
    py_types::{PyShape, Tensor, PY_UTILS},
    pycall, sv,
    tensor_util::{TorchDevice, TorchDeviceDtype, TorchDtype},
};
#[pyclass]
pub struct CharTokenizer {
    #[pyo3(get, set)]
    start: i64,
    #[pyo3(get, set)]
    end: i64,
    #[pyo3(get, set)]
    pad: i64,
    #[pyo3(get, set)]
    pad_width: i64,
    mapping: [i64; 256],
    error_if_over: bool,
}
#[pymethods]
impl CharTokenizer {
    #[new]
    pub fn new(
        start: i64,
        end: i64,
        pad: i64,
        pad_width: i64,
        mapping: HashMap<String, i64>,
        error_if_over: bool,
    ) -> Result<Self> {
        let mut new_mapping = [i64::MAX; 256];
        for (k, v) in mapping.into_iter() {
            if k.len() != 1 {
                bail!(anyhow!("char tokenizer token not 1 byte"))
            } else {
                new_mapping[k.bytes().next().unwrap() as usize] = v;
            }
        }
        Ok(Self {
            start,
            end,
            pad,
            pad_width,
            mapping: new_mapping,
            error_if_over,
        })
    }
    pub fn tokenize_strings(&self, strings: Vec<String>) -> Result<Tensor> {
        let mut strings = strings;
        let pad_width = self.pad_width as usize;
        let mut backing_array: Vec<i64> = vec![self.pad; pad_width * strings.len()];
        for (i, s) in strings.iter_mut().enumerate() {
            backing_array[i * pad_width + 0] = self.start;
            if self.error_if_over && s.len() > pad_width - 1 {
                bail!(anyhow!("string too wide {} {}", s.len(), pad_width));
            }
            s.truncate((pad_width - 2) as usize);
            backing_array[i * pad_width + s.len() + 1] = self.end;
            for (j, l) in s.bytes().enumerate() {
                if self.mapping[l as usize] == i64::MAX {
                    bail!(anyhow!("invalid char"))
                }
                backing_array[i * pad_width + j + 1] = self.mapping[l as usize];
            }
        }
        fn byteify<'a>(x: &'a [i64]) -> &'a [u8] {
            unsafe { std::slice::from_raw_parts(&x[0] as *const i64 as *const u8, x.len() * 8) }
        }

        let backing_array_bytes: &[u8] = byteify(&backing_array);

        pycall!(
            PY_UTILS.tensor_from_bytes,
            (
                TorchDeviceDtype {
                    dtype: TorchDtype::int64,
                    device: TorchDevice::Cpu
                },
                PyShape(sv![strings.len(), pad_width]),
                backing_array_bytes,
                backing_array.len()
            ),
            anyhow
        )
    }
}
