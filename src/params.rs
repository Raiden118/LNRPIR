use serde_json::Value;

use spiral_rs::{arith::*, params::*};

static DEFAULT_MODULI: [u64; 2] = [331777u64, 188417u64];
pub const DEF_MOD_STR: &str = "[\"331777\", \"188417\"]";

pub fn ext_params_from_json(json_str: &str) -> Params {
    let v: Value = serde_json::from_str(json_str).unwrap();

    let n = v["n"].as_u64().unwrap() as usize;
    let db_dim_1 = v["nu_1"].as_u64().unwrap() as usize;
    let db_dim_2 = v["nu_2"].as_u64().unwrap() as usize;
    let instances = v["instances"].as_u64().unwrap_or(1) as usize;
    let p: u64 = v["p"].as_u64().unwrap();
    let q2_bits = u64::max(v["q2_bits"].as_u64().unwrap(), MIN_Q2_BITS);
    let t_gsw = v["t_gsw"].as_u64().unwrap() as usize;
    let t_conv = v["t_conv"].as_u64().unwrap() as usize;
    let t_exp_left = v["t_exp_left"].as_u64().unwrap() as usize;
    let t_exp_right = v["t_exp_right"].as_u64().unwrap() as usize;
    let do_expansion = v.get("direct_upload").is_none();

    let mut db_item_size = v["db_item_size"].as_u64().unwrap_or(0) as usize;
    if db_item_size == 0 {
        db_item_size = instances * n * n;
        db_item_size = db_item_size * 2048 * log2_ceil(p) as usize / 8;
    }

    let version = v["version"].as_u64().unwrap_or(0) as usize;

    let poly_len = v["poly_len"].as_u64().unwrap_or(2048) as usize;
    let moduli = v["moduli"]
        .as_array()
        .map(|x| {
            x.as_slice()
                .iter()
                .map(|y| {
                    y.as_u64()
                        .unwrap_or_else(|| y.as_str().unwrap().parse().unwrap())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or(DEFAULT_MODULI.to_vec());
    let noise_width = v["noise_width"].as_f64().unwrap_or(6.4);

    Params::init(
        poly_len,
        &moduli,
        noise_width,
        n,
        p,
        q2_bits,
        t_conv,
        t_exp_left,
        t_exp_right,
        t_gsw,
        do_expansion,
        db_dim_1,
        db_dim_2,
        instances,
        db_item_size,
        version,
    )
}

pub fn ext_small_params_from_json(json_str: &str) -> Params {
    let v: Value = serde_json::from_str(json_str).unwrap();

    let n = v["n"].as_u64().unwrap() as usize;
    let db_dim_1 = v["nu_1"].as_u64().unwrap() as usize;
    let db_dim_2 = v["nu_2"].as_u64().unwrap() as usize;
    let instances = v["instances"].as_u64().unwrap_or(1) as usize;
    let p: u64 = v["p"].as_u64().unwrap();
    let q2_bits = u64::max(v["q2_bits"].as_u64().unwrap(), MIN_Q2_BITS);
    let t_gsw = v["t_gsw"].as_u64().unwrap() as usize;
    let t_conv = v["t_conv"].as_u64().unwrap() as usize;
    let t_exp_left = v["t_exp_left"].as_u64().unwrap() as usize;
    let t_exp_right = v["t_exp_right"].as_u64().unwrap() as usize;
    let do_expansion = v.get("direct_upload").is_none();

    let mut db_item_size = v["db_item_size"].as_u64().unwrap_or(0) as usize;
    if db_item_size == 0 {
        db_item_size = instances * n * n;
        db_item_size = db_item_size * 2048 * log2_ceil(p) as usize / 8;
    }

    let version = v["version"].as_u64().unwrap_or(0) as usize;

    let poly_len = v["poly_len"].as_u64().unwrap_or(2048) as usize;
    let moduli = v["moduli"]
        .as_array()
        .map(|x| {
            x.as_slice()
                .iter()
                .map(|y| {
                    y.as_u64()
                        .unwrap_or_else(|| y.as_str().unwrap().parse().unwrap())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or(DEFAULT_MODULI.to_vec());
    let noise_width = v["noise_width"].as_f64().unwrap_or(6.4);

    Params::init_small(
        poly_len,
        &moduli,
        noise_width,
        n,
        p,
        q2_bits,
        t_conv,
        t_exp_left,
        t_exp_right,
        t_gsw,
        do_expansion,
        db_dim_1,
        db_dim_2,
        instances,
        db_item_size,
        version,
    )
}

pub fn internal_params_for(
    poly_len : usize,
    nu_1: usize,
    nu_2: usize,
    p: u64,
    q2_bits: usize,
    t_exp_left: usize,
    moduli: &str,
) -> Params {     
    ext_params_from_json(&format!(
        r#"
        {{
            "poly_len": {},
            "n": 1,
            "nu_1": {},
            "nu_2": {},
            "p": {},
            "q2_bits": {},
            "t_gsw": 3,
            "t_conv": 5,
            "t_exp_left": {},
            "t_exp_right": 2,
            "instances": 1,
            "db_item_size": 0,
            "moduli": {},
            "noise_width": 16.042421
        }}
        "#,
        poly_len, nu_1, nu_2, p, q2_bits, t_exp_left, moduli
    ))
}

pub fn internal_small_params_for(
    poly_len : usize,
    nu_1: usize,
    nu_2: usize,
    p: u64,
    q2_bits: usize,
    t_exp_left: usize,
    moduli: &str,
) -> Params {     
    ext_small_params_from_json(&format!(
        r#"
        {{
            "poly_len": {},
            "n": 1,
            "nu_1": {},
            "nu_2": {},
            "p": {},
            "q2_bits": {},
            "t_gsw": 3,
            "t_conv": 5,
            "t_exp_left": {},
            "t_exp_right": 2,
            "instances": 1,
            "db_item_size": 0,
            "moduli": {},
            "noise_width": 16.042421
        }}
        "#,
        poly_len, nu_1, nu_2, p, q2_bits, t_exp_left, moduli
    ))
}
