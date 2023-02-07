use std::os::unix::prelude::OsStrExt;

use anyhow::{bail, Context, Result};
use base16::encode_lower;
use pyo3::prelude::*;

use crate::py_types::{Tensor, PY_UTILS};
pub fn get_rrfs_dir() -> String {
    std::env::var("RRFS_DIR").unwrap_or_else(|_e| std::env::var("HOME").unwrap() + "/rrfs")
}

pub fn get_tensor_by_hash_dir() -> String {
    std::env::var("TENSORS_BY_HASH_DIR")
        .unwrap_or_else(|_| get_rrfs_dir() + "/circuit_tensors_by_hash")
}

#[pyfunction]
pub fn tensor_from_hash(hash_base16: &str) -> Result<Tensor> {
    let hashdir = get_tensor_by_hash_dir() + "/" + hash_base16 + ".pt";
    let mut t: Tensor = Python::with_gil(|py| {
        PY_UTILS
            .torch
            .getattr(py, "load")
            .unwrap()
            .call(py, (hashdir,), None)
            .context("Failed to load tensor from hash")?
            .extract(py)
            .context("Failed to extract pyobject in tensor from hash")
    })?;

    if std::env::var("TENSORS_BY_HASH_REHASH_ON_LOAD").is_err() {
        t.set_hash(Some(
            ::base16::decode(hash_base16).unwrap().try_into().unwrap(),
        ));
    }

    Ok(t)
}

#[pyfunction]
pub fn tensor_from_hash_prefix(hash_base16: &str) -> Result<Tensor> {
    let hash_base16_bytes = hash_base16.as_bytes();
    let dir: Vec<_> = std::fs::read_dir(get_tensor_by_hash_dir())
        .unwrap()
        .into_iter()
        .filter(|x| {
            let nm = x.as_ref().unwrap().file_name();
            let name_bytes = nm.as_bytes();
            name_bytes.len() >= hash_base16_bytes.len()
                && &name_bytes[0..hash_base16_bytes.len()] == hash_base16_bytes
        })
        .collect();
    if dir.len() > 1 {
        bail!("tensor hash prefix ambiguous");
    }
    if dir.is_empty() {
        bail!("tensor from hash prefix not found {}", hash_base16);
    }
    tensor_from_hash(
        dir[0]
            .as_ref()
            .unwrap()
            .file_name()
            .to_str()
            .unwrap()
            .strip_suffix(".pt")
            .unwrap(),
    )
}

#[pyfunction]
pub fn save_tensor_rrfs(tensor: Tensor) -> Result<String> {
    let tensor = tensor.hashed()?;
    let hash_base16 = encode_lower(tensor.hash().unwrap());
    let hashdir = get_tensor_by_hash_dir() + "/" + &hash_base16 + ".pt";
    Python::with_gil(|py| {
        PY_UTILS
            .torch
            .getattr(py, "save")
            .context("save tensor get save attribute")?
            .call(py, (tensor.tensor(), hashdir), None)
            .context("save tensor")
            .map(|_| hash_base16)
    })
}
