use std::{fmt::Debug, hash::Hash, ptr};

use anyhow::Result;
use pyo3::{prelude::*, types::IntoPyDict};
use rustc_hash::FxHashMap as HashMap;

use crate::{py_types::Tensor, tensor_db::get_tensor_prefix, tensor_util::TorchDeviceDtype};
// implement our own weighted LRU cache bc didnt see a weighted library
// based on https://github.com/jeromefroe/lru-rs
#[derive(Debug, Clone)]
struct LruEntry<K: Default + Debug, V: Debug + Clone> {
    key: K,
    val: Option<V>,
    weight: usize,
    prev: *mut LruEntry<K, V>,
    next: *mut LruEntry<K, V>,
}
impl<K: Default + Debug, V: Debug + Clone> LruEntry<K, V> {
    fn new(key: K, val: V, weight: usize) -> Self {
        LruEntry {
            key,
            val: Some(val),
            weight,
            prev: ptr::null_mut(),
            next: ptr::null_mut(),
        }
    }

    fn new_sigil() -> Self {
        LruEntry {
            key: Default::default(),
            val: None,
            weight: 0,
            prev: ptr::null_mut(),
            next: ptr::null_mut(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LruCache<K: Eq + Hash + Default + Clone + Debug, V: Debug + Clone> {
    map: HashMap<K, Box<LruEntry<K, V>>>,
    cap: usize,
    current_weight: usize,

    // head and tail are sigil nodes to facilitate inserting entries
    head: *mut LruEntry<K, V>,
    tail: *mut LruEntry<K, V>,
}

impl<K: Eq + Hash + Default + Clone + Debug, V: Debug + Clone> LruCache<K, V> {
    pub fn new(cap: usize) -> Self {
        let cache = LruCache {
            map: HashMap::default(),
            cap,
            current_weight: 0,
            head: Box::into_raw(Box::new(LruEntry::new_sigil())),
            tail: Box::into_raw(Box::new(LruEntry::new_sigil())),
        };

        unsafe {
            (*cache.head).next = cache.tail;
            (*cache.tail).prev = cache.head;
        }

        cache
    }
    pub fn drop_last(&mut self) {
        if self.map.len() > 0 {
            unsafe {
                let entry = (*self.tail).prev;
                (*(*entry).prev).next = self.tail;
                (*self.tail).prev = (*entry).prev;
                self.current_weight -= (*entry).weight;
                self.map.remove(&(*entry).key);
            }
        }
    }
    pub fn insert(&mut self, k: K, v: V, weight: usize) {
        if self.get(&k).is_none() {
            // this get bumps to front
            self.current_weight += weight;
            let entry = Box::new(LruEntry::new(k.clone(), v, weight));
            self.map.insert(k.clone(), entry);
            while self.current_weight > self.cap {
                self.drop_last()
            }
            let mut entry = self.map.get_mut(&k).unwrap();
            unsafe {
                let entry_ptr: *mut LruEntry<K, V> = &mut **entry;
                entry.prev = self.head;
                entry.next = (*self.head).next;
                (*(*self.head).next).prev = entry_ptr;
                (*self.head).next = entry_ptr;
            }
        }
    }

    pub fn get(&mut self, k: &K) -> Option<&V> {
        self.map.get_mut(k).map(|entry| {
            unsafe {
                let entry_ptr: *mut LruEntry<K, V> = &mut **entry;
                let old_head = (*self.head).next;
                let old_entry_next = entry.next;
                let old_entry_prev = entry.prev;

                (*old_entry_prev).next = old_entry_next;
                (*old_entry_next).prev = old_entry_prev;
                entry.next = old_head;
                entry.prev = self.head;
                (*self.head).next = entry_ptr;
                (*old_head).prev = entry_ptr;
            }
            entry.val.as_ref().unwrap()
        })
    }

    pub fn delete(&mut self, k: &K) {
        if let Some(entry) = self.map.remove(k) {
            unsafe {
                (*entry.prev).next = entry.next;
                (*entry.next).prev = entry.prev;
            }
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct TensorCacheRrfs {
    size_cutoff: usize,
    small: LruCache<String, Tensor>,
    large: LruCache<String, Tensor>,
    device: String,
}
#[pymethods]
impl TensorCacheRrfs {
    #[new]
    pub fn new(
        cutoff: usize,
        small_capacity: usize,
        large_capacity: usize,
        device: String,
    ) -> Self {
        TensorCacheRrfs {
            size_cutoff: cutoff,
            small: LruCache::new(small_capacity),
            large: LruCache::new(large_capacity),
            device,
        }
    }

    pub fn get_tensor(&mut self, prefix: String) -> Result<Tensor> {
        self.small
            .get(&prefix)
            .or_else(|| self.large.get(&prefix))
            .map(|z| Ok(z.clone()))
            .unwrap_or_else(|| {
                let tensor = get_tensor_prefix(&prefix)?;
                let tensor: Tensor = Python::with_gil(|py| {
                    Ok::<Tensor, PyErr>(
                        tensor
                            .tensor()
                            .getattr(py, "to")?
                            .call(
                                py,
                                (),
                                Some(
                                    &[("device".to_owned(), self.device.clone())].into_py_dict(py),
                                ),
                            )?
                            .extract(py)?,
                    )
                })?;
                let size = tensor.shape().iter().cloned().product::<usize>()
                    * TorchDeviceDtype::from_tensor(&tensor).size();
                if size > self.size_cutoff {
                    self.large.insert(prefix, tensor.clone(), size)
                } else {
                    self.small.insert(prefix, tensor.clone(), size)
                }
                Ok(tensor)
            })
    }

    pub fn get_tensor_if_cached(&mut self, prefix: String) -> Option<Tensor> {
        self.small
            .get(&prefix)
            .or_else(|| self.large.get(&prefix))
            .map(|z| z.clone())
    }
}

#[test]
fn test_lru_cache() {
    let mut cache: LruCache<usize, usize> = LruCache::new(10);
    cache.insert(0, 5, 5);
    dbg!(&cache);
    cache.insert(1, 5, 5);
    dbg!(&cache);
    cache.insert(2, 5, 5);
    dbg!(&cache);
    cache.get(&1);
    cache.insert(3, 5, 5);
    dbg!(&cache);
    cache.insert(4, 2, 2);
    cache.insert(5, 2, 2);
    dbg!(&cache);
    cache.insert(6, 9, 9);
    dbg!(&cache);
    // cache.insert(4, 2, 2);
}
