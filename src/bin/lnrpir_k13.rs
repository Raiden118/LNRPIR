use clap::Parser;

use lnrpir::params::*;
use lnrpir::database::*;
use lnrpir::query::*;
use lnrpir::kernel::*;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use spiral_rs::aligned_memory::*;
use spiral_rs::{
    client::*,
    params::*,
    poly::*,
    gadget::*,
};

use spiral_rs::poly::{PolyMatrix, PolyMatrixRaw, PolyMatrixNTT};

use spiral_rs::arith::*;  

use std::time::Instant;

use rayon::prelude::*;

pub const RGSW_SEEDS: [[u8; 32]; 28] = [[7; 32]; 28];

pub const KSK_2_SEED: [u8; 32]= [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8];

pub const STATIC_PUBLIC_SEED: [u8; 32] = [0u8; 32];

pub fn perform_offline_precomputation<'a>(
       lwe_params:&'a Params,
       ntru_params:&'a Params,
       lwe_large_params:&'a Params,
       ksk1_a:&[PolyMatrixRaw<'a>],
       ksk2_a:&[PolyMatrixRaw<'a>],     
       db:&[u8],
       k:usize,
       Num:usize,
       num_layer_cmux:usize,
       t:usize,
) -> (Vec<PolyMatrixRaw<'a>>, Vec<Vec<PolyMatrixRaw<'a>>>) {
    let mut vec_a = Vec::new();
    let mut seed = STATIC_PUBLIC_SEED;
    for _i in 0..(1<<k) {
        let mut rng_pub = ChaCha20Rng::from_seed(seed);
        let a = PolyMatrixRaw::random_rng(lwe_params, 1, 1, &mut rng_pub);
        let arr = a.data.as_slice();
        vec_a.extend(arr.iter().map(|x| *x as u32));
        seed[0]=seed[0].wrapping_add(1);
        if seed[0]==0 {seed[1]+=1;}
    }
    let db_cols = 1<<k;
    let db_rows = Num/db_cols;
    let n = lwe_params.poly_len;
    let ell = lwe_params.t_exp_left;
    let modulus = ntru_params.modulus;
    let N = ntru_params.poly_len;
//println!("ell={}", ell);    
    let mut vec_poly_full = Vec::new();

println!("Doing offline part 1!");    
let batch_size = 64;
for batch_start in (0..db_rows).step_by(batch_size) {
    let batch_end = (batch_start + batch_size).min(db_rows);

    let batch_results: Vec<PolyMatrixRaw<'_>> =
        (batch_start..batch_end)
            .into_par_iter()
            .map(|r| {
                let db_one_row =
                    &db[r * db_cols..(r + 1) * db_cols];

                let mut vec_a_j = vec![0u32; db_cols];
                let mut intermediate_a = vec![0u64; n];
   
                for j in 0..n {
                    for c in 0..db_cols {
                        vec_a_j[c] = vec_a[c * n + j];
                    }

                    let packed_query_part =
                        pack_query_u32(&lwe_params, &vec_a_j);
                    
                    let mut intermediate = AlignedMemory32::new(1);

                    fast_batched_dot_product_u32_crt1(
                        &lwe_params,
                        intermediate.as_mut_slice(),
                        packed_query_part.as_slice(),
                        db_cols,
                        db_one_row,
                        db_cols,
                        1,
                    );                           
                
                    intermediate_a[j]=intermediate.as_slice()[0] as u64;
                }

                let mut sum =
                    PolyMatrixRaw::zero(ntru_params, 1, 1);

                for j in 0..n {      
                    let bits_per = get_bits_per(lwe_params, ell);                   
                    let base: u64 = 1u64 << bits_per;
                    let mask = base - 1;
                    let mut val = intermediate_a[j];
                    let mut gadget_digits = vec![0u64; ell];

                    for k in 0..ell {
                        gadget_digits[k] = val & mask;
                        val >>= bits_per;
                    }

                    let sca_mut_res =
                        fast_batched_scalar_product_no_reduce(   
                            &ksk1_a[j],
                            &gadget_digits,
                        );

                    fast_batched_add_raw(    
                        &mut sum,
                        &sca_mut_res,
                    );
                }
                for i in 0..N {
                    sum.get_poly_mut(0, 0)[i]= if sum.get_poly(0, 0)[i]==0 {0}
                                                        else { modulus - sum.get_poly(0, 0)[i] };
                }
                sum
            })
            .collect();
    vec_poly_full.extend(batch_results);
}

let N = ntru_params.poly_len;
    let vec_poly_a_part1: Vec<PolyMatrixRaw<'_>> = vec_poly_full
    .par_chunks_mut(N)
    .map(|chunk| {
        let mut sum = PolyMatrixRaw::zero(ntru_params, 1, 1);

        for (j, poly_matrix) in chunk.iter_mut().enumerate() {
            negacyclic_shift_inplace(
                poly_matrix.get_poly_mut(0, 0),
                j,
                modulus,
            );
  
            fast_batched_add_raw_no_reduce(&mut sum, poly_matrix);
        }

        fast_poly_raw_reduce(&mut sum);

        sum
    })
    .collect();
//println!("vec_poly_a_part1.len()={}", vec_poly_a_part1.len()); 

drop(vec_poly_full);  

println!("Doing offline part 2!");
    let mut vec_poly_a_part2 = Vec::new();   
    let mut seed = STATIC_PUBLIC_SEED;
    seed[2]+=1;
    let ell_prime = lwe_large_params.t_exp_left;
    let d = ntru_params.t_exp_left;
    for _t in 0..(num_layer_cmux+t) { 
        let mut temp_vec =Vec::new();   
        for i in 0..d {         
            let mut rng_pub = ChaCha20Rng::from_seed(seed);
            let a = PolyMatrixRaw::random_rng(&lwe_large_params, 1, 1, & mut rng_pub);
            let mut sum = PolyMatrixRaw::zero(&ntru_params, 1, 1);
            for j in 0..N {

                let gadget_digits = gadget_invert_scalar(
                       a.get_poly(0, 0)[j],
                        lwe_large_params,
                        ell_prime,
                    );    

                let mut sca_mut_res =
                        fast_batched_scalar_product_i64_no_reduce(
                            &ksk2_a[j],           
                            &gadget_digits,           
                        );

                for i in 0..N {
                    if sca_mut_res.get_poly(0, 0)[i] > u64::MAX/2 { 
                       let x = sca_mut_res.get_poly(0, 0)[i] as i64;
                       let r = x.rem_euclid(modulus as i64) as u64;
                       sca_mut_res.get_poly_mut(0, 0)[i] = r; };
                }

                fast_batched_add_raw_no_reduce(    
                        &mut sum,
                        &sca_mut_res,
                ); 
            }
            fast_poly_raw_reduce(&mut sum);

            for i in 0..N {
                    sum.get_poly_mut(0, 0)[i]= if sum.get_poly(0, 0)[i]==0 {0}
                                                        else { modulus - sum.get_poly(0, 0)[i] };
            }

            temp_vec.push(sum);
            seed[0]=seed[0].wrapping_add(1);
            if seed[0]==0 {seed[1]+=1;}
        }
    vec_poly_a_part2.push(temp_vec);
    }
println!("Done offline!");
(vec_poly_a_part1, vec_poly_a_part2)
}

pub fn run_lnrpir_on_params(     
    ntru_params: Params,
    lwe_params: Params,
    lwe_large_params: Params,
    rlwe_full_len_full_mod_params: Params,
    rlwe_full_len_small_mod_params: Params,
    rlwe_small_len_small_mod_params: Params,
    log_num_items: usize,   
    k: usize,
    trials: usize,
){
    let num_items= 1usize << log_num_items;   
    let N = ntru_params.poly_len;
    let log_N= ntru_params.poly_len_log2;
    let log_num = log_num_items;
//println!("log_num = {}", log_num);      
    let p= ntru_params.pt_modulus;
//println!("pt_modulus = {}", p);    
    let Q = ntru_params.modulus;  
//println!("Q = {}", Q);
    let n = lwe_params.poly_len;    
    let q= lwe_params.modulus;
//println!("q = {}", q);    
    let delta = q / p;
    let Delta = Q / q;
    let d = ntru_params.t_exp_left;
//println!("d={}", d);    
    let log_B = get_bits_per(&ntru_params, d);
//println!("log_B={}", log_B);    
//println!("ntru_crt_count={}", ntru_params.crt_count);    
    let db_rows = num_items / (1<<k);
    let db_cols = 1<<k;
    let n_1 = rlwe_full_len_full_mod_params.poly_len;
    let n_2 = rlwe_small_len_small_mod_params.poly_len;
    let t = n_1/n_2;
//println!("t={}", t);    
    let q_1 = rlwe_full_len_full_mod_params.modulus;
//println!("q_1={}", q_1);    
    let q_2 = rlwe_small_len_small_mod_params.modulus;
    let q_3 = 251u64;
//println!("t={}", t);     
    type T = u8;  

println!("Creating Database...");
    let mut pt_iter = std::iter::repeat_with(|| (T::sample() as u64 % p) as T);   
    let mut actual_db_origin = Vec::with_capacity(num_items);  
    for _ in 0..num_items {
        actual_db_origin.push(pt_iter.next().unwrap());   
    }
    
println!("Sample indices for all trials...");
    let (positions, samples) = sample_database(&actual_db_origin, n_2, trials + 1, k);   
    let actual_db = database_transform(&actual_db_origin, db_rows, db_cols, n_1, n_2);
    drop(actual_db_origin);

   let mut monomial_ntts = Vec::new();
        let mut neg_monomial_ntts = Vec::new();
        for j in 0..ntru_params.poly_len {
            let mut monomial = PolyMatrixRaw::zero(&ntru_params, 1, 1);
            monomial.get_poly_mut(0, 0)[j] = 1;
            let mono_ntt = monomial.ntt();
            monomial_ntts.push(mono_ntt.clone());
            neg_monomial_ntts.push(-&mono_ntt);
        }
    let mut rotation_db = PolyMatrixNTT::zero(&ntru_params, 1, t);
        for i in 0..t{
            let exponent = if i==0 {0} else {2 * ntru_params.poly_len - i};    
            if exponent < ntru_params.poly_len {
                rotation_db.get_poly_mut(0, i).copy_from_slice(monomial_ntts[exponent].as_slice());
            } else {
                rotation_db.get_poly_mut(0, i).copy_from_slice(neg_monomial_ntts[exponent-ntru_params.poly_len].as_slice());
            }
        }

//=====================================================================================================================================
println!("Generating keys...");
    let mut lwe_client = Client::init(&lwe_params);
    lwe_client.generate_secret_keys();  
    let sk_reg = lwe_client.get_sk_reg();  

    let mut ntru_client = Client::init(&ntru_params);
    ntru_client.generate_secret_key_and_inversion_ntru();

    let f= ntru_client.sk_ntru;   
    let f_inv = ntru_client.sk_ntru_inv;

    let mut lwe_large_client = Client::init(&lwe_large_params);
    lwe_large_client.generate_secret_keys();
    let sk_large = lwe_large_client.get_sk_reg();

    let mut rlwe_full_len_full_mod_client = Client::init(&rlwe_full_len_full_mod_params);
    rlwe_full_len_full_mod_client.generate_secret_keys(); 
    let sk_1 = rlwe_full_len_full_mod_client.get_sk_reg(); 

    let mut rlwe_small_len_small_mod_client = Client::init(&rlwe_small_len_small_mod_params);
    rlwe_small_len_small_mod_client.generate_secret_keys();
    let sk_2 = rlwe_small_len_small_mod_client.get_sk_reg();

    let ell = lwe_params.t_exp_left;     
    let log_z = get_bits_per(&lwe_params, ell);   
//println!("In KSK1, ell = {}, log_z = {}", ell, log_z);
    let mut ksk1_b_vec= Vec::new();
    let mut temp_k = PolyMatrixRaw::zero(&ntru_params, 1, 1);
        let mut rng = ChaCha20Rng::from_entropy();
        let mut g = PolyMatrixRaw::fast_noise(&ntru_params, 1, 1, &ntru_client.dg, &mut rng);
        let temp= (g.get_poly(0, 0)[0] + Delta) % Q;
        g.get_poly_mut(0, 0)[0] = temp;
        let temp_ct = (&g.ntt()*&f_inv.ntt()).raw();
        temp_k.get_poly_mut(0, 0).copy_from_slice(temp_ct.get_poly(0, 0));
    
    ksk1_b_vec.push(temp_k.clone());
    for i in 1..N {
        let mut monoial = PolyMatrixRaw::zero(&ntru_params, 1, 1);
        monoial.get_poly_mut(0, 0)[i]=1;   
        let temp_prime = (&monoial.ntt()*&temp_k.ntt()).raw();   
        ksk1_b_vec.push(temp_prime);
    }

    let mut ksk1_a = Vec::new();
    for j in 0..n {
        let mut temp_k = PolyMatrixRaw::zero(&ntru_params, 1, ell);
        for i in 0..ell {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut g = PolyMatrixRaw::fast_noise(&ntru_params, 1, 1, &ntru_client.dg, &mut rng);
        let term_i = if sk_reg.get_poly(0, 0)[j]>=q/2 { Q-(1<<(i*log_z))*Delta*(q-sk_reg.get_poly(0, 0)[j]) }   
                  else { (1<<(i*log_z))*Delta*sk_reg.get_poly(0, 0)[j] }; 
        let temp= (g.get_poly(0, 0)[0] + term_i) % Q;    
        g.get_poly_mut(0, 0)[0] = temp;
        let temp_ct = (&g.ntt()*&f_inv.ntt()).raw();    
        temp_k.get_poly_mut(0, i).copy_from_slice(temp_ct.get_poly(0, 0));
    }
   
    ksk1_a.push(temp_k);
    }
    
    let ell_prime = lwe_large_params.t_exp_left;
    let log_z_prime = get_bits_per(&lwe_large_params, ell_prime);
//println!("ell_prime={}, log_z_prime={}", ell_prime, log_z_prime);    
    let mut ksk2_b = PolyMatrixRaw::zero(&ntru_params, 1, ell_prime);
    for i in 0..ell_prime {
        let mut rng = ChaCha20Rng::from_entropy();
        let g = PolyMatrixRaw::fast_noise(&ntru_params, 1, 1, &ntru_client.dg, &mut rng);
        let mut temp = (&g.ntt()*&f_inv.ntt()).raw();    
        temp.get_poly_mut(0, 0)[0] = (temp.get_poly(0, 0)[0] + (1<<(i*log_z_prime))) % Q;
        ksk2_b.get_poly_mut(0, i).copy_from_slice(temp.get_poly(0, 0));
    }

    let mut ksk2_a = Vec::new();
    for j in 0..N {
        let mut temp_k = PolyMatrixRaw::zero(&ntru_params, 1, ell_prime);
        for i in 0..ell_prime {
        let mut rng = ChaCha20Rng::from_entropy();
        let g = PolyMatrixRaw::fast_noise(&ntru_params, 1, 1, &ntru_client.dg, &mut rng);
        let mut temp = (&g.ntt()*&f_inv.ntt()).raw();
        temp.get_poly_mut(0, 0)[0] = ((temp.get_poly(0, 0)[0] as u128 + (1u128<<(i*log_z_prime))*sk_large.get_poly(0, 0)[j] as u128) % (Q as u128)) as u64;    
        temp_k.get_poly_mut(0, i).copy_from_slice(temp.get_poly(0, 0));
    }
    ksk2_a.push(temp_k);
    }

    let ell_1=rlwe_full_len_full_mod_params.t_exp_left;    
    let log_z1=get_bits_per(&rlwe_full_len_full_mod_params, ell_1);
//println!("ell_1={}, log_z1={}", ell_1, log_z1);    
    let mut seed = STATIC_PUBLIC_SEED;
    seed[3]+=1;
    let mut ksk_rlwe = PolyMatrixRaw::zero(&rlwe_full_len_full_mod_params, 2, ell_1);
    for j in 0..ell_1 {
    let mut rng = ChaCha20Rng::from_entropy();
    let mut rng_pub = ChaCha20Rng::from_seed(seed);
    let a = PolyMatrixRaw::random_rng(&rlwe_full_len_full_mod_params, 1, 1, &mut rng_pub);
    let e = PolyMatrixRaw::fast_noise(&rlwe_full_len_full_mod_params, 1, 1, &rlwe_full_len_full_mod_client.dg, &mut rng);
    let a_s = (&a.ntt()*&sk_1.ntt()).raw();
    for i in 0..n_1 {
        let temp = (a_s.get_poly(0, 0)[i] + (1u64<<(j*log_z1))*f.get_poly(0, 0)[i] + e.get_poly(0, 0)[i]) % q_1;
        ksk_rlwe.get_poly_mut(1, j)[i] = temp;
    }

    ksk_rlwe.get_poly_mut(0, j).copy_from_slice(a.get_poly(0, 0));
    seed[0]=seed[0].wrapping_add(1);
    if seed[0]==0 {seed[1]+=1;}
    }

        let ell_2 = rlwe_small_len_small_mod_params.t_exp_left;
//println!("ell_2={}", ell_2);        
        let mut b_vec = Vec::new();  
        let mut ksk_seed = KSK_2_SEED;
        for _i in 0..t {
            let a = PolyMatrixNTT::random_rng(
                &rlwe_small_len_small_mod_params,
                1,
                ell_2,
                &mut ChaCha20Rng::from_seed(ksk_seed),
            );
            let e = PolyMatrixRaw::fast_noise(&rlwe_small_len_small_mod_params, 1, ell_2, &rlwe_small_len_small_mod_client.dg, &mut ChaCha20Rng::from_entropy());         
            let b = &(&sk_2.ntt() * &a);       
            b_vec.push(&b.raw() + &e);   
            ksk_seed[0] += 1;
        }
          
        let mut b_pack = PolyMatrixRaw::zero(&rlwe_full_len_small_mod_params, 1, ell_2);

        for j in 0..ell_2 {
            let result_poly = b_pack.get_poly_mut(0, j);
            for i in 0..t {
                let input_poly = b_vec[i].get_poly(0, j);
                for k in 0..n_2 {
                    result_poly[k * t + i] = input_poly[k];
                }
            }
        }          
        let log_z_2 = get_bits_per(&rlwe_small_len_small_mod_params, ell_2); 
//println!("log_z_2 = {}", log_z_2);             
        let mut w_2 = PolyMatrixRaw::zero(&rlwe_full_len_small_mod_params, 1, ell_2); 

        for i in 0..ell_2 {
            let shift = 1 << (log_z_2 * i);  
            let w_2_poly = w_2.get_poly_mut(0, i);
            let b_poly = b_pack.get_poly(0, i);
            let sk_1_poly = sk_1.get_poly(0, 0);  
            for k in 0..b_poly.len() {              
                let temp = if sk_1_poly[k] >= (rlwe_full_len_full_mod_params.modulus / 2) {q_2 + sk_1_poly[k] - rlwe_full_len_full_mod_params.modulus} else { sk_1_poly[k] };   
                let factor = (temp * shift) % q_2;    
                w_2_poly[k] = (b_poly[k] + q_2 - factor) % q_2;
            } 
        }
        let w_2_ntt = w_2.ntt();

//=====================================================================================================================================
println!("Offline computation...");
    let offline_comp_time = Instant::now();
    let num_layer_cmux = log_num_items-log_N-k;

    let (vec_poly_a_part1_origin, vec_poly_a_part2) = perform_offline_precomputation(
                                                   &lwe_params,
                                                   &ntru_params,
                                                   &lwe_large_params,
                                                   &ksk1_a,
                                                   &ksk2_a,
                                                   &actual_db,
                                                   k,
                                                   num_items,
                                                   num_layer_cmux,
                                                   t,
                                                   );
    let offline_comp_time_ms = offline_comp_time.elapsed().as_millis();
    println!("Offline computation time = {}ms", offline_comp_time_ms);
//=====================================================================================================================================
for trial in 0..trials + 1 {
println!("=============================================================");
println!("This is trial {}!", trial);    
println!("Generating query...");
    let mut vec_poly_a_part1 = vec_poly_a_part1_origin.clone();

    let target_index = positions[trial];
    let cols = 1<<k; 
    let target_group_row = target_index / cols;   
    let target_col = target_index % cols;                   
    let target_packed_row = target_group_row / t; 
    let target_rot = target_group_row % t;        
//println!("target_group_row={}, target_col={}, target_packed_row={}, target_rot={}", target_group_row, target_col, target_packed_row, target_rot);    

    let mut seed = STATIC_PUBLIC_SEED;
    let mut vec_b= Vec::new();
    for i in 0..(1<<k) {
        let mut rng_pub = ChaCha20Rng::from_seed(seed);
        let mut rng = ChaCha20Rng::from_entropy();
        let mut b=0;
        if i==target_col {b = generate_lwe_u32(&lwe_client, &mut rng, &mut rng_pub, delta as u32);}
           else {b = generate_lwe_u32(&lwe_client, &mut rng, &mut rng_pub, 0);}
        vec_b.push(b);
        seed[0]=seed[0].wrapping_add(1);
        if seed[0]==0 {seed[1]+=1;}
    }
//println!("query_row length = {}", vec_b.len());
    
    let packed_query_part_one = pack_query_u32(&lwe_params, &vec_b);  

    let num_layer_cmux = log_num_items-log_N-k;
//println!("num_layer_cmux = {}", num_layer_cmux);    
    let bits = decompose_u64_to_binary_coeffs(target_packed_row as u64, num_layer_cmux);
//println!("bits = {:?}", bits);    
    let mut query_part_two = Vec::new();
    let mut seed = STATIC_PUBLIC_SEED; 
    seed[2]+=1;
    
    for (_, pt) in bits.iter().enumerate() {  
        for i in 0..d { 
                let mut rng = ChaCha20Rng::from_entropy(); 
                let mut rng_pub = ChaCha20Rng::from_seed(seed);   
                let temp = generate_lwe_u64(&lwe_large_client, &mut rng, &mut rng_pub, (*pt as u64)*(1<<(i*log_B)));
                query_part_two.push(temp); 
                //seed[0] += 1;
                seed[0]=seed[0].wrapping_add(1);
                if seed[0] == 0 {seed[1] += 1}; 
                }
        }

    let mut bits = Vec::new();
    for i in 0..t {
        if i == target_rot {bits.push(1);}
        else {bits.push(0);}
    }

    for (_, pt) in bits.iter().enumerate() {   
            for i in 0..d {     
                let mut rng = ChaCha20Rng::from_entropy(); 
                let mut rng_pub = ChaCha20Rng::from_seed(seed);  
                let temp = generate_lwe_u64(&lwe_large_client, &mut rng, &mut rng_pub, (*pt as u64)*(1<<(i*log_B))); 
                query_part_two.push(temp); 
                seed[0]=seed[0].wrapping_add(1);
                if seed[0] == 0 {seed[1] += 1}; 
            }
        }
//println!("query_part_two len = {}, num_layer_cmux={}, t={}", query_part_two.len(), num_layer_cmux, t);
//==============================================================================================================       
println!("Doing online computation...");
    let lwe_pt_ct_time = Instant::now();
    let mut intermediate = AlignedMemory32::new(db_rows);
        fast_batched_dot_product_u32_crt1(   
            &lwe_params,
            intermediate.as_mut_slice(),
            packed_query_part_one.as_slice(),
            db_cols,
            &actual_db,
            db_cols,
            db_rows,
        );
    let lwe_pt_ct_time_ms = lwe_pt_ct_time.elapsed().as_millis();
    println!("Mult db time ={}ms", lwe_pt_ct_time_ms);

println!("Doing packing for the first part!");
    let packing_part_one_time = Instant::now();
    let arr_b: Vec<u64>= intermediate.as_slice().iter().map(|&x| x as u64).collect();
    for i in (0..arr_b.len()).step_by(N) {
        for j in 0..N {
        let sca_mut_res = fast_batched_scalar_product_no_reduce(&ksk1_b_vec[j], &arr_b[i+j..i+j+1]);
        fast_batched_add_raw_no_reduce(&mut vec_poly_a_part1[i/N], &sca_mut_res);
        if (j + 1) % 512 == 0 {
            fast_poly_raw_reduce(&mut vec_poly_a_part1[i / N]);
        }
        }
    }
    let packing_part_one_time_ms = packing_part_one_time.elapsed().as_millis();   
    println!("Packing part one time = {}ms", packing_part_one_time_ms);

assert!(query_part_two.len()==d*(num_layer_cmux+t));
let packing_part_two_time = Instant::now();
let mut vntru_vec = Vec::new();
for i in 0..(num_layer_cmux+t) {
    let mut vntru = PolyMatrixRaw::zero(&ntru_params, 1, d);
    for j in 0..d {
       
        let gadget_digits = gadget_invert_scalar(
                       query_part_two[i*d+j],                       
                        &lwe_large_params,
                        ell_prime,
                    );
        let mut sca_mut_res = fast_batched_scalar_product_i64_no_reduce(&ksk2_b, &gadget_digits);
                for i in 0..N {
                    if sca_mut_res.get_poly(0, 0)[i] > u64::MAX/2 { 
                       let x = sca_mut_res.get_poly(0, 0)[i] as i64;
                       let r = x.rem_euclid(Q as i64) as u64;
                       sca_mut_res.get_poly_mut(0, 0)[i] = r; };
                }
        fast_batched_add_raw(&mut sca_mut_res, &vec_poly_a_part2[i][j]);
        vntru.get_poly_mut(0, j).copy_from_slice(sca_mut_res.get_poly(0, 0));
    }
    vntru_vec.push(vntru); 
}
let packing_part_two_time_ms = packing_part_two_time.elapsed().as_millis();
println!("Packing part two time = {}ms", packing_part_two_time_ms);  

    println!("Doing Cmux...");      
    let cmux_time = Instant::now();
    let mut len = vec_poly_a_part1.len();  

    for j in 0..num_layer_cmux {   
        for i in 0.. len/2 {
                let (left, right) = vec_poly_a_part1.split_at_mut(2 * i + 1);
                ntru_cmux_raw( &mut left[2*i], &mut right[0], &vntru_vec[j].ntt());
            vec_poly_a_part1.swap(i, 2*i);
        }
        len /= 2;
    }
    let cmux_time_ms = cmux_time.elapsed().as_millis();

    assert_eq!(len, 1);
    println!("cmux time = {}ms", cmux_time_ms);
    let first = vec_poly_a_part1
    .into_iter()
    .next()
    .expect("packed_part_one must not be empty");
    vec_poly_a_part1 = vec![first];  
    
    
    println!("Doing rotation and switching...");
    let remaining_time = Instant::now();
    let mut temp = PolyMatrixRaw::zero(&ntru_params, t, d);
    for i in 0..t {
        for j in 0..d {
            let poly=vntru_vec[i+num_layer_cmux].get_poly(0, j);
            temp.get_poly_mut(i, j).copy_from_slice(poly);
        }
    }
    let rotation_ntt = &rotation_db*&temp.ntt();

    let mut sub_decomp = PolyMatrixRaw::zero(&ntru_params, d, 1);
    let mut sub_decomp_ntt = PolyMatrixNTT::zero(&ntru_params, d, 1);

    gadget_invert(&mut sub_decomp, &vec_poly_a_part1[0]);     
    to_ntt(&mut sub_decomp_ntt, &sub_decomp);

    let mut prod = PolyMatrixNTT::zero(&ntru_params, 1, 1);
    multiply(&mut prod, &rotation_ntt, &sub_decomp_ntt);

    let ntru_after_rot = prod.raw();


    let mut sub_decomp = PolyMatrixRaw::zero(&rlwe_full_len_full_mod_params, ell_1, 1);
    let mut sub_decomp_ntt = PolyMatrixNTT::zero(&rlwe_full_len_full_mod_params, ell_1, 1);
    gadget_invert(&mut sub_decomp, &ntru_after_rot);    
    to_ntt(&mut sub_decomp_ntt, &sub_decomp);
    let mut prod = PolyMatrixNTT::zero(&rlwe_full_len_full_mod_params, 2, 1);
    multiply(&mut prod, &ksk_rlwe.ntt(), &sub_decomp_ntt);
    let rlwe = prod.raw();

    println!("Doing compression...");       
    let mut ksk_seed = KSK_2_SEED;
        let mut a_vec = Vec::new();
        for _i in 0..t {
            let a = PolyMatrixNTT::random_rng(
                &rlwe_small_len_small_mod_params,
                1,
                ell_2,
                &mut ChaCha20Rng::from_seed(ksk_seed),
            );           
            a_vec.push(a.raw());   
            ksk_seed[0] += 1;
        }
        let mut w_1 = PolyMatrixRaw::zero(&rlwe_full_len_small_mod_params, 1, ell_2);

        for j in 0..ell_2 {

            let result_poly = w_1.get_poly_mut(0, j);
    
            for i in 0..t {
                let input_poly = a_vec[i].get_poly(0, j);

                for k in 0..n_2 {
                    result_poly[k * t + i] = input_poly[k];
                }
            }
        }
        let mut prod = PolyMatrixNTT::zero(&rlwe_full_len_small_mod_params, 1, 1);    
        let mut temp_c1 = PolyMatrixRaw::zero(&rlwe_full_len_small_mod_params, 1, 1);  
        let input_poly = rlwe.get_poly(0, 0);  
        let result_poly = temp_c1.get_poly_mut(0, 0);

        for k in 0..input_poly.len() {
            result_poly[k] = rescale(input_poly[k], rlwe_full_len_full_mod_params.modulus, q_2);  
        }       
        let mut gad_inv = PolyMatrixRaw::zero(&rlwe_full_len_small_mod_params, ell_2, 1); 
        let mut gad_inv_ntt = PolyMatrixNTT::zero(&rlwe_full_len_small_mod_params, ell_2, 1);
        gadget_invert(&mut gad_inv, &temp_c1);  
                to_ntt(&mut gad_inv_ntt, &gad_inv);  
                multiply(        
                    &mut prod,
                    &w_1.ntt(),  
                    &gad_inv_ntt,
                );        
        let c1_prime = prod.raw();       
        let mut c1_hat = PolyMatrixRaw::zero(&rlwe_small_len_small_mod_params, 1, 1);
        let input_poly = c1_prime.get_poly(0, 0);
        let output_poly = c1_hat.get_poly_mut(0, 0);
        for i in 0..rlwe_small_len_small_mod_params.poly_len {
            output_poly[i] = input_poly[t*i];    
        }
        gadget_invert(&mut gad_inv, &temp_c1);  
                to_ntt(&mut gad_inv_ntt, &gad_inv);  
                multiply(        
                    &mut prod,
                    &w_2_ntt,  
                    &gad_inv_ntt,
                );
        let prod_raw = prod.raw();   
        let mut temp_c2 = PolyMatrixRaw::zero(&rlwe_full_len_small_mod_params, 1, 1);  
        let input_poly = rlwe.get_poly(1, 0);  
        let result_poly = temp_c2.get_poly_mut(0, 0);
        for k in 0..input_poly.len() {
            result_poly[k] = rescale(input_poly[k], rlwe_full_len_full_mod_params.modulus, q_3);
        }
        let mut temp_c2_prime = PolyMatrixRaw::zero(&rlwe_full_len_small_mod_params, 1, 1);
        let input_poly = prod_raw.get_poly(0, 0);  
        let result_poly = temp_c2_prime.get_poly_mut(0, 0);
        for k in 0..input_poly.len() {
            result_poly[k] = rescale(input_poly[k], q_2, q_3); 
        }
        let temp = &temp_c2 + &temp_c2_prime;          
        let mut c2_hat = PolyMatrixRaw::zero(&rlwe_small_len_small_mod_params, 1, 1);  
        let input_poly = temp.get_poly(0, 0);
        let output_poly = c2_hat.get_poly_mut(0, 0);
        for i in 0..rlwe_small_len_small_mod_params.poly_len {
            output_poly[i] = input_poly[t*i] % (q_3); 
        }
        let mut final_rlwe = Vec::new(); 
        final_rlwe.push(c1_hat);
        final_rlwe.push(c2_hat);

        let remaining_time_ms = remaining_time.elapsed().as_millis();
        println!("rotation, switching and compression time = {}ms", remaining_time_ms);
        println!("total online time = {}ms", lwe_pt_ct_time_ms+packing_part_one_time_ms+packing_part_two_time_ms+cmux_time_ms+remaining_time_ms);
//=============================================================================================================================
println!("Checking response...");

    let temp = (&-(&sk_2.ntt()) * &(final_rlwe[0].ntt())).raw();
        let mut temp_prime = PolyMatrixRaw::zero(&rlwe_small_len_small_mod_params, 1, 1);
        let input_poly = temp.get_poly(0, 0);
        let output_poly = temp_prime.get_poly_mut(0, 0);
        for i in 0..rlwe_small_len_small_mod_params.poly_len {
            output_poly[i] = rescale(input_poly[i], q_2, q_3);
        }      
        let mut res = PolyMatrixRaw::zero(&rlwe_small_len_small_mod_params, 1, 1);  
        for i in 0..temp_prime.data.len() {
            res.data[i] = (final_rlwe[1].data[i] + temp_prime.data[i]) % (q_3); 
        }
        let mut pt = PolyMatrixRaw::zero(&rlwe_small_len_small_mod_params, 1, 1);
        let input_poly = res.get_poly(0, 0);
        let output_poly = pt.get_poly_mut(0, 0);
        for i in 0..rlwe_small_len_small_mod_params.poly_len {
            output_poly[i] = rescale(input_poly[i], q_3, rlwe_full_len_full_mod_params.pt_modulus);
        }
        let final_result: Vec<u8> = pt
                                    .as_slice()
                                    .iter()
                                    .map(|&x| x as u8)
                                    .collect();
        assert_eq!(final_result.len(), n_2);
        assert_eq!(final_result, samples[trial],"Something went wrong!");
println!("The response is correct!");
println!("=============================================================");
}

let log_Q = (Q as f64).log2();
let log_q = (q as f64).log2();
let log_q_1 = log_Q;
let log_q_2 = (q_2 as f64).log2();
let log_q_3 = (q_3 as f64).log2();
println!("k = {}", k);
let temp = ((1<<k) as f64)*log_q+(d as f64*(log_num_items - log_N - k + t) as f64)*log_Q;
let query_size = temp /(8.0*1024.0);
let response_size = (n_2 as f64) * (log_q_2 + log_q_3) /(8.0*1024.0);
let offline_key_size =((1+n*ell+(N+1)*ell_prime) as f64*(N as f64)*log_Q + (ell_1*n_1) as f64 *log_q_1 + (ell_2*n_1) as f64 *log_q_2)/(8.0*1024.0*1024.0);
println!("query_size is {}KB", query_size);
println!("response_size is {}KB", response_size);
println!("offline_key_size is {}MB", offline_key_size);
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of items in the database
    //#[clap(long)]
    //num_items: Option<usize>,

    //Logrithm of number of items in the database
    #[clap(long)]
    log_num_items: Option<usize>,

    /// Interpolation degree
    //#[clap(long)]
    //dim0: Option<usize>,

    /// Interpolation degree
    //#[clap(long)]
    //interpolate_degree: Option<usize>,

    //#[clap(long)]
    //item_size_bits: Option<usize>,

    /// Small params mode (optional)
    /// if set, small paramseters will be used
    //#[clap(long, short, action)]
    //small_params: bool,

    #[clap(long)]
    trials: Option<usize>,

    /// Output report file (optional)
    /// where results will be written in JSON.
    //#[clap(long)]
    //out_report_json: Option<String>,

    /// Just a label
    //#[clap(long)]
    //label: Option<String>,

    /// Verbose mode (optional)
    /// if set, the program will print debug logs to stderr.
    #[clap(long, short, action)]
    verbose: bool,

    //#[clap(long, short, action)]
    //online_only: bool,

}

fn main() {
    let args = Args::parse();
    let Args {
        //num_items,
        log_num_items,
        //dim0,
        //interpolate_degree,
        //item_size_bits,
        //small_params,
        trials,
        //out_report_json,
        //label,
        verbose,
        //online_only,
    } = args;

    if verbose {
        println!("Running in verbose mode.");
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .write_style(env_logger::WriteStyle::Always)
            .init();
    } else {
        env_logger::init();
    }

    let log_num_items = log_num_items.unwrap_or(31);   
    let num_items = 1u64<<log_num_items;

    let trials = trials.unwrap_or(5);   


    let k: usize = 13;              


    let poly_len_log2 = 11;        
    let poly_len = 1 << poly_len_log2; 
    let p = 16;
    let log_p = 4;
    let d = 4;
    pub const MOD_STR: &str =  "[\"331777\",\"188417\"]";   

    let mut ntru_params = internal_params_for(poly_len, k-poly_len_log2, 0, p, 28, d, MOD_STR);
    ntru_params.noise_width = 1.1;     


    let n=1024;
    pub const LWE_STR: &str = "[\"124339\"]";  
    let ell = 2;  
    let mut lwe_params = internal_small_params_for(n, 0, 0, p, 28, ell, LWE_STR);
    lwe_params.noise_width = 0.8;

  
    let ell_prime = 9;
    let mut lwe_large_params = internal_params_for(poly_len, 0, 0, p, 28, ell_prime, MOD_STR);
    lwe_large_params.noise_width = 0.8;


    pub const RLWE_STR: &str =  "[\"331777\",\"188417\"]";   
    let n_1= 2048;
    let ell_1=2;
    let mut rlwe_full_len_full_mod_params = internal_params_for(n_1, 0, 0, p, 28, ell_1, RLWE_STR);
    rlwe_full_len_full_mod_params.noise_width = 0.8;


    let n_2=512;
    let ell_2 = 11;    
    pub const RLWE_SMALL_STR: &str =  "[\"2056193\"]";    
    let mut rlwe_full_len_small_mod_params = internal_params_for(n_1, 0, 0, p, 28, ell_2, RLWE_SMALL_STR);
    rlwe_full_len_small_mod_params.noise_width = 0.8;


    let mut rlwe_small_len_small_mod_params = internal_params_for(n_2, 0, 0, p, 28, ell_2, RLWE_SMALL_STR); 
    rlwe_small_len_small_mod_params.noise_width=27.9;

    println!("DB Size={} GB", (num_items as f64 * log_p as f64)  / (8. * 1024. * 1024. * 1024.));

    run_lnrpir_on_params(ntru_params, 
                            lwe_params, 
                            lwe_large_params,
                            rlwe_full_len_full_mod_params,
                            rlwe_full_len_small_mod_params, 
                            rlwe_small_len_small_mod_params,
                            log_num_items, 
                            k, 
                            trials);

}

//8GB
//cargo run --release --bin lnrpir_k13 -- --log-num-items 34 > output_k13_8g.txt
