//! Detect whether an image is the cat "桃子" (Peach).
//!
//! The classifier is a ResNet-18 exported to ONNX; its weights are embedded into this
//! library at compile time (`include_bytes!`), so no external model file is needed at
//! runtime. The model is parsed and optimised lazily, once per process.
//!
//! # Example
//! ```no_run
//! let p = ispeach::predicate_path("cat.jpg")?;
//! if p >= ispeach::PEACH_THRESHOLD {
//!     println!("桃子! confidence={p}");
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, imageops::FilterType};
use tract_onnx::prelude::*;

/// Weights are fixed (embedded) into the library at compile time.
static MODEL_BYTES: &[u8] = include_bytes!("../model.onnx");

/// Confidence at or above which an image is considered "桃子".
pub const PEACH_THRESHOLD: f32 = 0.5;

const CROP: u32 = 224;
const RESIZE: u32 = 256;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A runnable, optimised ONNX model. `TypedRunnableModel` is `Arc<SimplePlan<..>>`.
type Model = Arc<TypedRunnableModel>;

fn build_model() -> Result<Model> {
    let model = tract_onnx::onnx()
        .model_for_read(&mut &MODEL_BYTES[..])
        .context("failed to parse embedded onnx model")?
        .with_input_fact(0, f32::fact([1, 3, CROP as usize, CROP as usize]).into())?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

/// Parse and optimise the embedded model once, then hand out a shared reference.
fn model() -> Result<&'static Model> {
    static CACHE: OnceLock<Result<Model, String>> = OnceLock::new();
    match CACHE.get_or_init(|| build_model().map_err(|e| format!("{e:#}"))) {
        Ok(m) => Ok(m),
        Err(e) => bail!("{e}"),
    }
}

/// Probability (0..=1) that the image is 桃子. Core entry point: accepts a decoded image.
pub fn predicate(image: &DynamicImage) -> Result<f32> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());

    // torchvision Resize(256): resize the shorter side to 256, keep aspect ratio.
    let scale = RESIZE as f32 / w.min(h) as f32;
    let nw = ((w as f32 * scale).round() as u32).max(CROP);
    let nh = ((h as f32 * scale).round() as u32).max(CROP);
    let resized = image::imageops::resize(&rgb, nw, nh, FilterType::Triangle);

    // torchvision CenterCrop(224).
    let x0 = (nw - CROP) / 2;
    let y0 = (nh - CROP) / 2;
    let crop = image::imageops::crop_imm(&resized, x0, y0, CROP, CROP).to_image();

    let mut input = tract_ndarray::Array4::<f32>::zeros((1, 3, CROP as usize, CROP as usize));
    for (x, y, px) in crop.enumerate_pixels() {
        for c in 0..3 {
            let v = px[c] as f32 / 255.0;
            input[[0, c, y as usize, x as usize]] = (v - MEAN[c]) / STD[c];
        }
    }

    let input: Tensor = input.into();
    let outputs = model()?.run(tvec!(input.into()))?;
    softmax_peach_prob(&outputs[0])
}

/// Convenience wrapper: read the image from a file path, then [`predicate`].
pub fn predicate_path<P: AsRef<Path>>(path: P) -> Result<f32> {
    let path = path.as_ref();
    let image =
        image::open(path).with_context(|| format!("failed to open image: {}", path.display()))?;
    predicate(&image)
}

/// Whether the image is 桃子 (confidence `>=` [`PEACH_THRESHOLD`]).
pub fn is_peach(image: &DynamicImage) -> Result<bool> {
    Ok(predicate(image)? >= PEACH_THRESHOLD)
}

/// Whether the image at `path` is 桃子.
pub fn is_peach_path<P: AsRef<Path>>(path: P) -> Result<bool> {
    Ok(predicate_path(path)? >= PEACH_THRESHOLD)
}

fn softmax_peach_prob(output: &TValue) -> Result<f32> {
    let logits = output
        .to_plain_array_view::<f32>()
        .context("unexpected output tensor type")?;
    let flat: Vec<f32> = logits.iter().copied().collect();
    if flat.len() != 2 {
        bail!("expected 2 output logits, got {}", flat.len());
    }
    let m = flat[0].max(flat[1]);
    let e0 = (flat[0] - m).exp();
    let e1 = (flat[1] - m).exp();
    Ok(e1 / (e0 + e1))
}

/// Returns the ISA features the underlying inference engine detects for this CPU
/// (e.g. `avx2`, `fma`). Useful to confirm SIMD dispatch. Best-effort: empty if unknown.
pub fn cpu_isa() -> Vec<String> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut out = Vec::new();
        for (name, present) in [
            ("avx", std::arch::is_x86_feature_detected!("avx")),
            ("avx2", std::arch::is_x86_feature_detected!("avx2")),
            ("fma", std::arch::is_x86_feature_detected!("fma")),
            ("f16c", std::arch::is_x86_feature_detected!("f16c")),
            ("avx512f", std::arch::is_x86_feature_detected!("avx512f")),
            (
                "avx512vnni",
                std::arch::is_x86_feature_detected!("avx512vnni"),
            ),
        ] {
            if present {
                out.push(name.to_string());
            }
        }
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec::new()
    }
}
