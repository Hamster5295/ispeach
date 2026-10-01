use std::process::exit;

use ispeach::{PEACH_THRESHOLD, cpu_isa, predicate_path};

fn run(path: &std::ffi::OsStr) {
    match predicate_path(path) {
        Ok(prob) => {
            // Confidence (probability of being 桃子) goes to stdout.
            println!("{prob:.6}");
            // Exit status conveys the decision: 0 = is 桃子, 1 = not 桃子.
            exit(if prob >= PEACH_THRESHOLD { 0 } else { 1 });
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            exit(2);
        }
    }
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    match args.next() {
        Some(a) if a.to_str() == Some("--isa") => {
            println!("{}", cpu_isa().join(","));
        }
        Some(a) if a.to_str() == Some("-h") || a.to_str() == Some("--help") => {
            eprintln!("usage: ispeach <img>   # stdout: confidence; exit 0=桃子 1=不是");
            eprintln!("       ispeach --isa   # print detected CPU ISA features");
        }
        Some(path) => run(&path),
        None => {
            eprintln!("usage: ispeach <img>\n       ispeach --isa");
            exit(2);
        }
    }
}
