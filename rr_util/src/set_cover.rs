use std::{cmp::Ordering, collections::BinaryHeap};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Entry {
    idx: usize,
    set: u128,
    count: u32,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn set_cover(sets: &Vec<u128>) -> Vec<usize> {
    let mut heap: BinaryHeap<Entry> = sets
        .iter()
        .enumerate()
        .map(|(i, x)| Entry {
            idx: i,
            set: *x,
            count: x.count_ones(),
        })
        .collect();
    let mut result = Vec::new();
    while heap.len() > 0 {
        let chosen = heap.pop().unwrap();
        result.push(chosen.idx);
        let mut new_heap = BinaryHeap::with_capacity(heap.len());
        for i in heap {
            let set = i.set & !chosen.set;
            let count = set.count_ones();
            if count > 0 {
                new_heap.push(Entry {
                    idx: i.idx,
                    set: set,
                    count: count,
                });
            }
        }
        heap = new_heap;
    }
    result
}

#[test]
fn test_set_cover() {
    let sets = vec![7u128, 3u128, 1u128, 8];
    dbg!(set_cover(&sets));
}
