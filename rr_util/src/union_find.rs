use rustc_hash::FxHashMap as HashMap;

pub struct UnionFind(Vec<usize>);

impl UnionFind {
    pub fn new(size: usize) -> Self {
        UnionFind((0..size).collect())
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        let p = &mut self.0;
        while x != p[x] {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }

    // find without path halving. use normal find if you can
    pub fn find_(&self, mut x: usize) -> usize {
        let p = &self.0;
        while x != p[x] {
            x = p[x];
        }
        return x;
    }

    // rem's union find https://algocoding.wordpress.com/2015/05/13/simple-union-find-techniques/
    pub fn union(&mut self, mut x: usize, mut y: usize) {
        let p = &mut self.0;
        loop {
            let p_x = p[x];
            let p_y = p[y];
            if p_x == p_y {
                break;
            } else if p_x < p_y {
                if x == p_x {
                    p[x] = p_y;
                    break;
                } else {
                    p[x] = p_y;
                    x = p_x;
                }
            } else {
                if y == p_y {
                    p[y] = p_x;
                    break;
                } else {
                    p[y] = p_x;
                    y = p_y;
                }
            }
        }
    }

    pub fn to_vec_vec(&mut self) -> Vec<Vec<usize>> {
        let mut dict: HashMap<usize, Vec<usize>> = HashMap::default();
        for i in 0..self.0.len() {
            dict.entry(self.find(i)).or_insert(Vec::new()).push(i);
        }
        dict.values().cloned().collect()
    }

    pub fn to_vec_vec_(&self) -> Vec<Vec<usize>> {
        let mut dict: HashMap<usize, Vec<usize>> = HashMap::default();
        for (i, x) in self.0.iter().enumerate() {
            dict.entry(self.find_(*x)).or_insert(Vec::new()).push(i);
        }
        dict.values().cloned().collect()
    }
}
