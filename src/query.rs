use rand_chacha::ChaCha20Rng;

use spiral_rs::aligned_memory::*;
use spiral_rs::{
    client::*,
    params::*,
};

use spiral_rs::poly::{PolyMatrix, PolyMatrixRaw};

pub fn pack_query_u32(params: &Params, query: &[u32]) -> AlignedMemory32 {
    let mut aligned_query_packed = AlignedMemory32::new(query.len());

    for (dst, x) in aligned_query_packed
        .as_mut_slice()
        .iter_mut()
        .zip(query.iter())
    {
        *dst = (*x as u64 % params.moduli[0]) as u32;
    }

    aligned_query_packed
}

pub fn generate_lwe_u32 (   
        client: &Client,         
        rng: &mut ChaCha20Rng,       
        rng_pub: &mut ChaCha20Rng,   
        pt: u32,     
    ) -> u32 {  
        let params = client.params;
        let a = PolyMatrixRaw::random_rng(params, 1, 1, rng_pub);  
        let mut b = 0u32;
        for i in 0..params.poly_len {
            b = ((b as u64 + (a.get_poly(0,0)[i] as u64)* (client.sk_reg.get_poly(0,0)[i] as u64)) % (params.modulus)) as u32
        }
        let e = client.dg.fast_sample(params.modulus, rng) as u32; 
        b = (b + pt + e) % (params.modulus as u32);
        b
    }

pub fn generate_lwe_u64 (        
        client: &Client,         
        rng: &mut ChaCha20Rng,       
        rng_pub: &mut ChaCha20Rng,   
        pt: u64,     
    ) -> u64 {  
        let params = client.params;
        let a = PolyMatrixRaw::random_rng(params, 1, 1, rng_pub);  
        let mut b = 0u64;
        for i in 0..params.poly_len {
            b = ((b as u128 + (a.get_poly(0,0)[i] as u128)* (client.sk_reg.get_poly(0,0)[i] as u128)) % (params.modulus as u128)) as u64
        }
        let e = client.dg.fast_sample(params.modulus, rng);         
        b = (b + pt + e) % (params.modulus);
        b
    }