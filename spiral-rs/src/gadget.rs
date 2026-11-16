use std::println;

use crate::{params::*, poly::*};

pub fn get_bits_per(params: &Params, dim: usize) -> usize {  //Default: modulus is NOT an exponent of 2.
    /* 
    let modulus_log2 = params.modulus_log2;
    if dim as u64 == modulus_log2 {
        return 1;
    }
    ((modulus_log2 as f64) / (dim as f64)).floor() as usize + 1
    */
    let modulus_log2 = params.modulus_log2;   
    if modulus_log2 as usize % dim == 0 { modulus_log2 as usize / dim }
    else { ((modulus_log2 as f64) / (dim as f64)).floor() as usize + 1 }
}

pub fn build_gadget(params: &Params, rows: usize, cols: usize) -> PolyMatrixRaw {
    let mut g = PolyMatrixRaw::zero(params, rows, cols);
    let nx = g.rows;
    let m = g.cols;

    assert_eq!(m % nx, 0);

    let num_elems = m / nx;
    let params = g.params;
    let bits_per = get_bits_per(params, num_elems);

    for i in 0..nx {
        for j in 0..num_elems {
            if bits_per * j >= 64 {
                continue;
            }
            let poly = g.get_poly_mut(i, i + j * nx);
            poly[0] = 1u64 << (bits_per * j);
        }
    }
    g
}

/*
pub fn gadget_invert_rdim<'a>(out: &mut PolyMatrixRaw<'a>, inp: &PolyMatrixRaw<'a>, rdim: usize) {
    assert_eq!(out.cols, inp.cols);

    let params = inp.params;
    let mx = out.rows;
    let num_elems = mx / rdim;
    let bits_per = get_bits_per(params, num_elems);
    let mask = (1u64 << bits_per) - 1;

    for j in 0..rdim {
        for i in 0..inp.cols {
            for z in 0..params.poly_len {
                let val = inp.get_poly(j, i)[z];
                for k in 0..num_elems {
                    let bit_offs = k * bits_per;
                    let piece = if bit_offs >= 64 {
                        0
                    } else {
                        (val >> bit_offs) & mask
                    };

                    //out.get_poly_mut(j + k * rdim, i)[z] = piece;
                    out.get_poly_mut(j * num_elems + k, i)[z] = piece; 
                }
            }
        }
    }
}
*/

 
//need centered gadget decomposition 
pub fn gadget_invert_rdim<'a>(       //q cannot be too large
    out: &mut PolyMatrixRaw<'a>,
    inp: &PolyMatrixRaw<'a>,
    rdim: usize,
) {
    assert_eq!(out.cols, inp.cols);
    let params = inp.params;
    let mx = out.rows;
    let num_elems = mx / rdim;
    let bits_per = get_bits_per(params, num_elems);   
    let base: i64 = 1i64 << bits_per;    
    let half_base = base >> 1;
    let q = params.modulus as i64;
    for j in 0..rdim {
        for i in 0..inp.cols {
            for z in 0..params.poly_len {

                let mut val =
                    inp.get_poly(j,i)[z] as i64;

                // 1. center lift
                if val >= q/2 {
                    val -= q;
                }            
                // 2. centered decomposition
                for k in 0..num_elems {
                    let mut digit =
                        val & (base-1);

                    // into signed digit
                    if digit >= half_base {
                        digit -= base;
                    }

                    // store as Z_q
                    let digit_mod_q =
                        if digit < 0 {
                            (digit + q) as u64
                        } else {
                            digit as u64
                        };

                    out.get_poly_mut(
                        j*num_elems+k,
                        i
                    )[z] = digit_mod_q;

                    val =
                        (val-digit)/base;

                }
            }
        }
    }
}

pub fn gadget_invert_scalar_with_rounding(
    inp: u64,
    params: &Params,
    num_elems: usize,
) -> Vec<u64> {
    let bits_per = get_bits_per(params, num_elems);
    let base: i64 = 1i64 << bits_per;
    let half_base = base >> 1;
    let q = params.modulus as i64;

    let mut val = inp as i64;

    // 1. Center lift
    if val >= q / 2 {
        val -= q;
    }

    // 2. Centered gadget decomposition, low digit first
    let mut out = vec![0u64; num_elems];

    for k in 0..num_elems {
        let mut digit = val & (base - 1);

        // Convert to centered digit
        if digit >= half_base {
            digit -= base;
        }

        // Store as Z_q
        out[k] = if digit < 0 {
            (digit + q) as u64
        } else {
            digit as u64
        };

        val = (val - digit) / base;
    }

    out
}

pub fn gadget_invert_scalar(
    inp: u64,
    params: &Params,
    num_elems: usize,
) -> Vec<i64> {
    let bits_per = get_bits_per(params, num_elems);
    let base: i64 = 1i64 << bits_per;
    let half_base = base >> 1;
    let q = params.modulus as i64;

    let mut val = inp as i64;

    // 1. Center lift
    if val >= q / 2 {
        val -= q;
    }

    // 2. Centered gadget decomposition, low digit first
    let mut out = vec![0i64; num_elems];

    for k in 0..num_elems {
        let mut digit = val & (base - 1);

        // Convert to centered digit
        if digit >= half_base {
            digit -= base;
        }

        // Store the signed digit directly
        out[k] = digit;

        val = (val - digit) / base;
    }
    out
}

pub fn gadget_invert<'a>(out: &mut PolyMatrixRaw<'a>, inp: &PolyMatrixRaw<'a>) {
    gadget_invert_rdim(out, inp, inp.rows);
}

pub fn gadget_invert_alloc<'a>(mx: usize, inp: &PolyMatrixRaw<'a>) -> PolyMatrixRaw<'a> {
    let mut out = PolyMatrixRaw::zero(inp.params, mx, inp.cols);
    gadget_invert(&mut out, inp);
    out
}

#[cfg(test)]
mod test {
    use crate::util::get_test_params;

    use super::*;

    #[test]
    fn gadget_invert_is_correct() {
        let params = get_test_params();
        let mut mat = PolyMatrixRaw::zero(&params, 2, 1);
        mat.get_poly_mut(0, 0)[37] = 3;
        mat.get_poly_mut(1, 0)[37] = 6;
        let log_q = params.modulus_log2 as usize;
        let result = gadget_invert_alloc(2 * log_q, &mat);

        assert_eq!(result.get_poly(0, 0)[37], 1);
        assert_eq!(result.get_poly(2, 0)[37], 1);
        assert_eq!(result.get_poly(4, 0)[37], 0); // binary for '3'

        assert_eq!(result.get_poly(1, 0)[37], 0);
        assert_eq!(result.get_poly(3, 0)[37], 1);
        assert_eq!(result.get_poly(5, 0)[37], 1);
        assert_eq!(result.get_poly(7, 0)[37], 0); // binary for '6'
    }
}
