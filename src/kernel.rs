use std::arch::x86_64::*;

use spiral_rs::{
    params::*,
    poly::*,
    gadget::*,
};

use spiral_rs::poly::{PolyMatrix, PolyMatrixRaw, PolyMatrixNTT};

use spiral_rs::arith::*;  

pub trait ToM512U32 {
        fn to_m512_u32(self) -> __m512i;
    }

#[cfg(target_feature = "avx512f")]
mod m512_u32_impl {
    use super::*;

    impl ToM512U32 for *const u8 {
        #[inline(always)]
        fn to_m512_u32(self) -> __m512i {
            unsafe {
                let x = _mm_loadu_si128(self as *const __m128i);

                _mm512_cvtepu8_epi32(x)
            }
        }
    }
}

struct AlignedU32Array16([u32; 16]);
pub fn fast_batched_dot_product_u32_crt1<T:Copy>(
    params: &Params,
    c: &mut [u32], 
    a: &[u32],     
    a_elems: usize,
    b_t: &[T], 
    b_rows: usize,
    b_cols: usize,
) where
*const T: ToM512U32, {
    assert_eq!(a_elems, b_rows);
    assert_eq!(a.len(), a_elems, "For K=1, 'a' should have a_elems elements.");
    assert_eq!(c.len(), b_cols, "For K=1, 'c' should have b_cols elements.");

    let simd_width = 16;            
    let k_chunk_size_factor = if simd_width > 0 {
        a_elems / simd_width
    } else {
        0
    };

    let chunk_size = (1024).min(k_chunk_size_factor);     
    let num_chunks = if chunk_size > 0 {
        k_chunk_size_factor / chunk_size
    } else {
        0
    };

    let j_num_chunks = b_cols;

    unsafe {
        let b_ptr = b_t.as_ptr();

        for k_outer in 0..num_chunks {
            for j_outer in 0..j_num_chunks {
                let mut total_sum = _mm512_setzero_si512();

                for k_inner in 0..chunk_size {
                    let k_base =
                        simd_width * (k_outer * chunk_size + k_inner);
                    let a_idx = k_base;
                    let b_idx = j_outer * b_rows + k_base;

                    let b_val_simd =
                        b_ptr.add(b_idx).to_m512_u32();

                    let a_val_simd = _mm512_loadu_si512(
                        a.as_ptr().add(a_idx) as *const __m512i,
                    );

                    let product =
                        _mm512_mullo_epi32(a_val_simd, b_val_simd);

                    total_sum =
                        _mm512_add_epi32(total_sum, product);
                }


                let mut values = AlignedU32Array16([0u32; 16]);
                _mm512_store_si512(
                    values.0.as_mut_ptr() as *mut __m512i,
                    total_sum,
                );

                let res = values.0.iter().map(|&x| x as u64).sum::<u64>();

                let mut lo_coeff = barrett_coeff_u32(params, res, 0);

                if k_outer!=0 { lo_coeff = barrett_u32(params, c[j_outer] as u64 + lo_coeff as u64,); }               
                
                c[j_outer] = lo_coeff;    
            }
        }
    }
}

pub fn fast_batched_scalar_product_no_reduce<'a>(        
    mat: &PolyMatrixRaw<'a>,
    scalars: &[u64],
) -> PolyMatrixRaw<'a> {
    assert_eq!(mat.rows, 1);
    assert_eq!(mat.cols, scalars.len());
    assert!(!scalars.is_empty());

    let n = mat.params.poly_len;
    let mut result = PolyMatrixRaw::zero(mat.params, 1, 1);

    unsafe {
        scalar_product_avx512(
            mat.data.as_slice(),
            scalars,
            result.get_poly_mut(0, 0),
            n,
        );
    }

    result
}

pub fn fast_batched_scalar_product_i64_no_reduce<'a>(             mat: &PolyMatrixRaw<'a>,
    scalars: &[i64],
) -> PolyMatrixRaw<'a> {
    assert_eq!(mat.rows, 1);
    assert_eq!(mat.cols, scalars.len());
    assert!(!scalars.is_empty());

    let n = mat.params.poly_len;
    let mut result = PolyMatrixRaw::zero(mat.params, 1, 1);

    unsafe {
        scalar_product_avx512_i64(
            mat.data.as_slice(),
            scalars,
            result.get_poly_mut(0, 0),
            n,
        );
    }

    result
}

#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn scalar_product_avx512(
    data: &[u64],
    scalars: &[u64],
    out: &mut [u64],
    n: usize,
) {
    for j in (0..n).step_by(8) {
        let mut acc = _mm512_setzero_si512();

        for i in 0..scalars.len() {
            let coeffs = _mm512_loadu_si512(
                data.as_ptr().add(i * n + j) as *const __m512i,
            );

            let scalar = _mm512_set1_epi64(scalars[i] as i64);
            let product = _mm512_mullo_epi64(coeffs, scalar);

            acc = _mm512_add_epi64(acc, product);
        }

        _mm512_storeu_si512(
            out.as_mut_ptr().add(j) as *mut __m512i,
            acc,
        );
    }
}

#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn scalar_product_avx512_i64(
    data: &[u64],
    scalars: &[i64],
    out: &mut [u64],
    n: usize,
) {
    for j in (0..n).step_by(8) {
        let mut acc = _mm512_setzero_si512();

        for i in 0..scalars.len() {
            let coeffs = _mm512_loadu_si512(
                data.as_ptr().add(i * n + j) as *const __m512i,
            );

            let scalar = _mm512_set1_epi64(scalars[i] as i64);
            let product = _mm512_mullo_epi64(coeffs, scalar);

            acc = _mm512_add_epi64(acc, product);
        }

        _mm512_storeu_si512(
            out.as_mut_ptr().add(j) as *mut __m512i,
            acc,
        );
    }
}

pub fn fast_batched_add_raw_no_reduce(
    lhs: &mut PolyMatrixRaw<'_>,
    rhs: &PolyMatrixRaw<'_>,
) {
    assert_eq!(lhs.rows, rhs.rows);
    assert_eq!(lhs.cols, rhs.cols);
    assert_eq!(lhs.params.poly_len, rhs.params.poly_len);

    let len = lhs.rows * lhs.cols * lhs.params.poly_len;

    unsafe {
        add_poly_matrix_raw_avx512(
            lhs.data.as_mut_slice(),
            rhs.data.as_slice(),
            len,
        );
    }
}

pub fn fast_batched_add_raw(
    lhs: &mut PolyMatrixRaw<'_>,
    rhs: &PolyMatrixRaw<'_>,
) {
    assert_eq!(lhs.rows, rhs.rows);
    assert_eq!(lhs.cols, rhs.cols);
    assert_eq!(lhs.params.poly_len, rhs.params.poly_len);

    let len = lhs.rows * lhs.cols * lhs.params.poly_len;

    unsafe {
        add_poly_matrix_raw_avx512(
            lhs.data.as_mut_slice(),
            rhs.data.as_slice(),
            len,
        );
    }

    let data = lhs.data.as_mut_slice();
    for i in 0..len {
        data[i] = barrett_u64(lhs.params, data[i]);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn add_poly_matrix_raw_avx512(
    lhs: &mut [u64],
    rhs: &[u64],
    len: usize,
) {
    for i in (0..len).step_by(8) {
        let a = _mm512_loadu_si512(
            lhs.as_ptr().add(i) as *const __m512i,
        );
        let b = _mm512_loadu_si512(
            rhs.as_ptr().add(i) as *const __m512i,
        );

        let sum = _mm512_add_epi64(a, b);

        _mm512_storeu_si512(
            lhs.as_mut_ptr().add(i) as *mut __m512i,
            sum,
        );
    }
}

pub fn fast_poly_raw_reduce(
    mat: &mut PolyMatrixRaw<'_>,
) {
    let len = mat.rows * mat.cols * mat.params.poly_len;
    let data = mat.data.as_mut_slice();
    let params = mat.params;
    for i in 0..len {
        data[i] = barrett_u64(params, data[i]);
    }
}

pub fn negacyclic_shift_inplace(a: &mut [u64], shift: usize, modulus: u64) {
    let n = a.len(); 
    assert!(shift<n);

    if shift == 0 {
        return;
    }

    a.rotate_right(shift);

    for x in a.iter_mut().take(shift) {
        assert!(*x<modulus);
        *x = if *x == 0 { 0 } else { modulus - *x };
    }
}

pub fn decompose_u64_to_binary_coeffs(to_decompose: u64, bit_length: usize) -> Vec<u8> {    
    let mut n = to_decompose;
    assert!(n<(1<<bit_length)); 

    let mut coeffs = vec![0u8; bit_length]; 

    for i in 0..bit_length {
        if n == 0 {break;}
        coeffs[i]=(n & 1) as u8;
        n >>= 1;       
    }
    coeffs 
}

pub fn ntru_cmux_raw<'a>(
    c_0: &mut PolyMatrixRaw<'a>,          
    c_1: &mut PolyMatrixRaw<'a>,      
    vntru_b: & PolyMatrixNTT<'a>,      
) {
    let params=c_0.params;
    let d = params.t_exp_left;   

    for i in 0..c_0.rows {
        for j in 0..c_1.cols {
            let pol0 = c_0.get_poly(i, j);
            let pol1 = c_1.get_poly_mut(i, j);
                for l in 0..params.poly_len {
                    let a = pol1[l];
                    let b = pol0[l];
                    pol1[l] = if a >= b {
                                   a - b
                                }
                                else {
                                a + params.modulus - b
                                };
                }
        }
    }

    let mut sub_decomp = PolyMatrixRaw::zero(params, d, 1);
    let mut sub_decomp_ntt = PolyMatrixNTT::zero(params, d, 1);

    gadget_invert(&mut sub_decomp, &c_1);      
    to_ntt(&mut sub_decomp_ntt, &sub_decomp);
    let mut prod = PolyMatrixNTT::zero(params, 1, 1);
    multiply(&mut prod, &vntru_b, &sub_decomp_ntt);  
    *c_0 += &prod.raw(); 
}
