use std::{
    cmp::min,
    fs,
    io::Write,
    iter::zip,
    os::unix::prelude::OsStrExt,
    sync::{mpsc::channel, Mutex},
};

use anyhow::{anyhow, bail, Context, Result};
use base16::encode_lower;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use rustc_hash::FxHashSet as HashSet;
use threadpool::ThreadPool;

use crate::{
    py_types::{Tensor, PY_UTILS},
    rrfs::get_rrfs_dir,
    util::HashBytes,
};

static TENSOR_CACHE_DIR: Lazy<String> = Lazy::new(|| {
    std::env::var("TENSORS_CACHE_DIR")
        .unwrap_or(std::env::var("HOME").unwrap() + &"/tensors_by_hash_cache")
});
static UNSYNCED_LIST: Lazy<String> = Lazy::new(|| TENSOR_CACHE_DIR.clone() + "/.unsynced_list");

pub fn get_rrfs_tensor_db_dir() -> String {
    get_rrfs_dir() + "/tensor_db"
}

const SECTIONS: [usize; 3] = [4, 4, 56];
const SECTION_STARTS: [usize; 3] = [0, 4, 8];

pub fn get_tensor_hash_parts(tensor: &Tensor) -> Vec<String> {
    let b16 = encode_lower(&tensor.hashed().unwrap().hash().unwrap());
    zip(&SECTION_STARTS, &SECTIONS)
        .map(|(a, b)| b16[*a..*a + *b].to_owned())
        .collect()
}

pub fn split_prefix(prefix: &str) -> (Vec<String>, String) {
    let n_full_sections = min(
        SECTIONS.len() - 1,
        (0..SECTION_STARTS.len())
            .map(|i| (SECTION_STARTS[i] + SECTIONS[i] <= prefix.len()) as usize)
            .sum(),
    );

    (
        (0..n_full_sections)
            .map(|i| prefix[SECTION_STARTS[i]..SECTION_STARTS[i + 1]].to_owned())
            .collect(),
        prefix[SECTION_STARTS[n_full_sections]..].to_owned(),
    )
}

pub fn write_tensor_to_dir_tree(dir: &str, tensor: Tensor, force: bool) -> Result<bool> {
    let hash_parts = get_tensor_hash_parts(&tensor);
    let dir = dir.to_owned()
        + "/"
        + &hash_parts[..SECTIONS.len() - 1]
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("/");
    fs::create_dir_all(&dir)?;
    let filename = dir + "/" + hash_parts.last().unwrap();
    if !force && fs::metadata(&filename).is_ok() {
        return Ok(false);
    }
    Python::with_gil(|py| {
        PY_UTILS
            .torch
            .getattr(py, "save")
            .context("save tensor get save attribute")?
            .call(py, (tensor.tensor(), filename), None)
            .context("save tensor")
    })?;
    Ok(true)
}

pub fn get_only_file_in_nested_dirs(dir: &str) -> Result<String> {
    let mut dir = dir.to_owned();
    loop {
        let staty = fs::metadata(&dir).context("cant find dir in tree")?;
        if staty.is_file() {
            return Ok(dir);
        }
        if staty.is_dir() {
            let files: Vec<_> = fs::read_dir(&dir)?.collect();
            if files.len() > 1 {
                bail!(anyhow!("prefix dir contains multiple things"));
            }
            if files.len() == 0 {
                bail!(anyhow!("dir has no files"));
            }
            dir = dir + "/" + &files[0].as_ref().unwrap().file_name().to_str().unwrap();
        }
    }
}

pub fn get_tensor_dir_tree(dir: &str, hash_prefix: &str) -> Result<Tensor> {
    get_filename_dir_tree(dir, hash_prefix).and_then(|(s, h)| load_tensor(s, Some(h)))
}

pub fn get_filename_dir_tree(dir: &str, hash_prefix: &str) -> Result<(String, String)> {
    let (wholes, part) = split_prefix(&hash_prefix);
    let look_dir = dir.to_owned() + "/" + &wholes.join("/");
    let files = fs::read_dir(&look_dir)
        .with_context(|| format!("tensors dir doesnt exist {}", &look_dir))?;
    let matching_entries: Vec<_> = files
        .filter(|x| &x.as_ref().unwrap().file_name().as_bytes()[..part.len()] == part.as_bytes())
        .collect();
    match matching_entries.len() {
        0 => {
            bail!(anyhow!("Found no tensors matching hash prefix"))
        }
        1 => {
            let longpath = look_dir
                + "/"
                + &matching_entries[0]
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_str()
                    .unwrap();
            if matching_entries[0]
                .as_ref()
                .unwrap()
                .file_type()
                .unwrap()
                .is_file()
            {
                Ok(longpath)
            } else {
                get_only_file_in_nested_dirs(&longpath)
            }
            .map(|file| (file.clone(), file[dir.len()..].replace("/", "")))
        }
        _ => {
            bail!(anyhow!("Found multiple tensors matching hash prefix"))
        }
    }
}

pub fn load_tensor(dir: String, hash: Option<String>) -> Result<Tensor> {
    Python::with_gil(|py| {
        PY_UTILS
            .torch
            .getattr(py, "load")
            .unwrap()
            .call(py, (&dir, "cpu"), None)
            .context(format!("Failed to load tensor from hash {}", &dir))?
            .extract(py)
            .context("Failed to extract pyobject in tensor from hash")
            .and_then(|t: Tensor| {
                let Some(hash) = hash else {
                    return Ok(t);
                };
                let hash_vec: Vec<u8> = base16::decode(&hash).context("hash not base16")?;
                let hash: HashBytes = HashBytes::try_from(hash_vec)
                    .map_err(|e| anyhow!("wrong length hash {}", e.len()))?;
                let mut t = t;
                t.set_hash(Some(hash));
                Ok(t)
            })
    })
}

pub fn register_tensor_unsynced(hash: &str) -> Result<()> {
    writeln!(
        fs::OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(&*UNSYNCED_LIST)
            .unwrap(),
        "{}",
        hash
    )
    .context("couldnt write to unsynced list")
}

#[pyfunction(signature=(tensor, force = false))]
pub fn save_tensor(tensor: Tensor, force: bool) -> Result<bool> {
    fs::create_dir_all(&*TENSOR_CACHE_DIR)?;

    let result = write_tensor_to_dir_tree(&*TENSOR_CACHE_DIR, tensor.clone(), force);
    if force{
        write_tensor_to_dir_tree(&get_rrfs_tensor_db_dir(), tensor.clone(), force)?;
    }else if let Ok(did_write) = result && did_write{
        register_tensor_unsynced(&tensor.hashed()?.hash_base16().unwrap())?;
    }
    result
}

fn copy_without_setting_permissions(local_filename: &str, remote_filename: &str) -> Result<()> {
    let mut reader = fs::File::open(local_filename)
        .context(format!("opening {} in read mode failed", local_filename))?;
    let mut writer = fs::File::create(remote_filename)
        .context(format!("opening {} in write mode failed", remote_filename))?;
    std::io::copy(&mut reader, &mut writer).context("writing contents failed")?;
    Ok(())
}

/// this is only useful bc querying multiple tensors in parallel is much faster than in series
pub fn ensure_all_tensors_local(hashes: Vec<String>) -> Result<()> {
    static THREAD_POOL: Lazy<Mutex<ThreadPool>> = Lazy::new(|| {
        Mutex::new({
            let n_workers = 30;
            ThreadPool::new(n_workers)
        })
    });
    // println!("loading {} tensors", hashes.len());
    let nitems = hashes.len();
    let (tx, rx) = channel();
    {
        let pool = THREAD_POOL.lock().unwrap();
        for hash in hashes {
            let tx = tx.clone();
            pool.execute(move || {
                let result = ensure_prefix_local(&hash).map(|_| ());
                // println!("did");
                tx.send(result).unwrap()
            });
        }
    }
    rx.iter().take(nitems).collect::<Result<()>>()
}

#[pyfunction(signature=(parallelism = 15))]
pub fn sync_all_unsynced_tensors(parallelism: usize) -> Result<()> {
    if !fs::try_exists(&*UNSYNCED_LIST)? {
        println!(
            "didnt find any unsynced list (~/tensors_by_hash_cache/.unsynced_list), returning"
        );
        return Ok(());
    }
    println!("starting to read list");
    let all_unsynced = fs::read_to_string(&*UNSYNCED_LIST)?;
    println!("read list {}", all_unsynced.len());
    let lines: Vec<String> = all_unsynced
        .lines()
        .filter(|x| !x.is_empty())
        .map(|x| x.to_owned())
        .collect();

    sync_specific_tensors(lines, parallelism)
}

#[pyfunction(signature=(tensor_hash_prefixes, parallelism = 15))]
pub fn sync_specific_tensors(tensor_hash_prefixes: Vec<String>, parallelism: usize) -> Result<()> {
    // we can have more threads than cores bc we're not actually doing computation on threads
    // ideally would be using async
    let n_workers = parallelism;
    let pool = ThreadPool::new(n_workers);
    let (tx, rx) = channel();
    for hash_prefix_str in &tensor_hash_prefixes {
        let hash_prefix_str = hash_prefix_str.clone();
        let tx = tx.clone();
        pool.execute(move || {
            tx.send((move || {
                let local_filename = get_filename_dir_tree(&*TENSOR_CACHE_DIR, &hash_prefix_str)?.0;
                let remote_filename =
                    local_filename.replace(&*TENSOR_CACHE_DIR, &get_rrfs_tensor_db_dir());
                let remote_filename_dir: Vec<_> = remote_filename.split("/").collect();
                let remote_filename_dir = remote_filename_dir
                    [..remote_filename_dir.len().checked_add_signed(-1).unwrap()]
                    .join("/");
                fs::create_dir_all(&remote_filename_dir).context("failed to create dir in tree")?;
                copy_without_setting_permissions(&local_filename, &remote_filename)
                    .context("failed to copy")?;
                eprintln!("wrote {}", hash_prefix_str);
                Ok::<(), anyhow::Error>(())
            })())
            .unwrap()
        });
    }
    rx.iter()
        .take(tensor_hash_prefixes.len())
        .collect::<Result<Vec<()>>>()?;

    if !fs::try_exists(&*UNSYNCED_LIST)? {
        println!(
            "didnt find any unsynced list (~/tensors_by_hash_cache/.unsynced_list), returning"
        );
        return Ok(());
    }

    let all_unsynced = fs::read_to_string(&*UNSYNCED_LIST)?;
    println!("read list {}", all_unsynced.len());
    let still_unsynced_lines: Vec<String> = all_unsynced
        .lines()
        // removing all with prefix is fine because we check prefix only matched one above
        // (Up to race condition on matching prefix - we probably should have some sort of lock etc!)
        .filter(|x| !x.is_empty() && !tensor_hash_prefixes.iter().any(|p| x.starts_with(p))) // note any is inefficient, but who cares...
        .map(|x| x.to_owned() + "\n")
        .collect();

    fs::write(&*UNSYNCED_LIST, still_unsynced_lines.join("")).context("write failed")
}

#[pyfunction]
pub fn get_tensor_prefix(prefix: &str) -> Result<Tensor> {
    get_tensor_dir_tree(&*TENSOR_CACHE_DIR, &prefix).or_else(|_e| {
        let result = get_tensor_dir_tree(&get_rrfs_tensor_db_dir(), &prefix);
        if let Ok(t) = &result {
            write_tensor_to_dir_tree(&*TENSOR_CACHE_DIR, t.clone(), false)?;
        }
        result
    })
}

#[pyfunction]
pub fn ensure_prefix_local(prefix: &str) -> Result<()> {
    get_filename_dir_tree(&*TENSOR_CACHE_DIR, &prefix)
        .map(|_| ())
        .or_else(|_e| {
            let remote_filename = get_filename_dir_tree(&get_rrfs_tensor_db_dir(), prefix)?.0;
            let local_filename =
                remote_filename.replace(&get_rrfs_tensor_db_dir(), &*TENSOR_CACHE_DIR);
            let local_filename_dir: Vec<_> = local_filename.split("/").collect();
            let local_filename_dir = local_filename_dir
                [..local_filename_dir.len().checked_add_signed(-1).unwrap()]
                .join("/");
            fs::create_dir_all(&local_filename_dir).context("failed to create dir in tree")?;
            copy_without_setting_permissions(&remote_filename, &local_filename)
                .context("failed to copy")?;
            Ok(())
        })
}

#[pyfunction]
pub fn migrate_tensors_from_old_dir(dir: &str) -> Result<()> {
    let files: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .map(|x| x.unwrap().file_name().to_str().unwrap().to_owned())
        .collect();
    print!("have list");
    let all_unsynced = fs::read_to_string(&*UNSYNCED_LIST)?;
    let lines: HashSet<_> = all_unsynced[..all_unsynced.len() - 1].lines().collect();

    for file in files {
        if !lines.contains(&&file[..64]) {
            print!("O");
            let tensor = load_tensor(dir.to_owned() + "/" + &file, None)?;
            save_tensor(tensor, false)?;
        } else {
            print!("_");
        }
    }
    Ok(())
}
