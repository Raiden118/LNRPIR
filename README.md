# LNRPIR: High Throughput PIR with Flexible Communication-Computation Trade-offs

This is an implementation of the LNRPIR protocol for Private Information Retrieval (PIR).
This scheme was introduced in the paper "LNRPIR: High Throughput PIR with Flexible Communication-Computation Trade-offs".

## Installation

1. Install Rust via rustup:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```
2. Clone the repository
`git clone <repository-url>`,
then
`cd <repository-directory>`

3. Run `cargo build`.

Stability warnings can be safely ignored. 

## Requirements
- Ubuntu (Linux).
- CPU with AVX-512 support (`avx512f` flag in `lscpu`).
- Rust toolchain (installed according to the Installation section).

## Running
To run the experiment over a 64 MB database, run:
```bash
cargo run --release --bin lnrpir_k9 -- --log-num-items 27 > output_k9_64mb.txt
cargo run --release --bin lnrpir_k10 -- --log-num-items 27 > output_k10_64mb.txt
cargo run --release --bin lnrpir_k11 -- --log-num-items 27 > output_k11_64mb.txt
```

To run the experiment over a 256 MB database, run:
```bash
cargo run --release --bin lnrpir_k10 -- --log-num-items 29 > output_k10_256mb.txt
cargo run --release --bin lnrpir_k11 -- --log-num-items 29 > output_k11_256mb.txt
cargo run --release --bin lnrpir_k12 -- --log-num-items 29 > output_k12_256mb.txt
```

To run the experiment over a 1 GB database, run:
```bash
cargo run --release --bin lnrpir_k10 -- --log-num-items 31 > output_k10_1g.txt
cargo run --release --bin lnrpir_k11 -- --log-num-items 31 > output_k11_1g.txt
cargo run --release --bin lnrpir_k12 -- --log-num-items 31 > output_k12_1g.txt
```

To run the experiment over an 8 GB database, run:
```bash
cargo run --release --bin lnrpir_k12 -- --log-num-items 34 > output_k12_8g.txt
cargo run --release --bin lnrpir_k13 -- --log-num-items 34 > output_k13_8g.txt
cargo run --release --bin lnrpir_k14 -- --log-num-items 34 > output_k14_8g.txt
```
Every command will execute 5 actual trials by default (the extra trial 0 is the warm-up run and should be excluded). Output includes:
- The database size.
- Online time for different parts of computation and the total online time for every trial.
- Parameter k used in the setting of this experiment.
- Query size, response size, and offline communication in this setting.
