use rand::Rng;

pub trait Sample {
    fn sample() -> Self;
}

impl Sample for u8 {
    fn sample() -> Self {
        fastrand::u8(..)
    }
}

pub fn sample_database(                    
    db: &[u8],
    n: usize,        
    trial: usize,
    k:usize,
) -> (Vec<usize>, Vec<Vec<u8>>) {
    let num_entries = db.len() / n;   
    let cols = 1<<k;
    let mut rng = rand::thread_rng();
    let mut positions = Vec::with_capacity(trial);    
    let mut samples = Vec::with_capacity(trial);
    for _ in 0..trial {
        let t = rng.gen_range(0..num_entries);
        positions.push(t);
        
        let idx_group_row = t / cols;  
        let idx_col = t % cols;        
        let start_row = idx_group_row * n;
        let value = (0..n)
            .map(|r| db[(start_row + r) * cols + idx_col])    
            .collect();

        samples.push(value);
    }
    (positions, samples)
}

pub fn database_transform(
    db:&[u8],
    rows:usize,
    cols:usize,
    n_1:usize,
    n_2:usize,
) -> Vec<u8> {
    let length = db.len();
    let mut output = vec![0u8; length];
    let t = n_1/n_2;
    for i in 0..rows {
        for j in 0.. cols {
            let num_group_n1 = i/n_1;   
            let num_in_group_n1 = i%n_1;  
            let num_group_n2 = num_in_group_n1/n_2;   
            let num_in_group_n2 = num_in_group_n1%n_2;  
            output[(num_group_n1*n_1+num_group_n2+num_in_group_n2*t)*cols+j]=db[i*cols+j];
        }
    }
    output
}