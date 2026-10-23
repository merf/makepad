//! A probe, not a check of the stream (RationalVisions docs/cutie-metal.md, MPSGraph): one of
//! Cutie's convolutions in fp16 on MPSGraph, its input and output our own MTLBuffers, against an
//! f64 reference, and its time. `cargo test -p makepad-ai-metal --release --test mpsgraph_probe -- --nocapture`
#![cfg(target_os = "macos")]

use makepad_objc_sys::runtime::ObjcId;
use makepad_objc_sys::{class, msg_send, sel, sel_impl};
use std::ffi::c_void;
use std::time::Instant;

#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> ObjcId;
}
#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}
#[link(name = "MetalPerformanceShadersGraph", kind = "framework")]
extern "C" {}
#[link(name = "Foundation", kind = "framework")]
extern "C" {}

const MPS_FLOAT16: u32 = 0x1000_0000 | 16;
const MPS_FLOAT32: u32 = 0x1000_0000 | 32;
/// MPSGraphPaddingStyleExplicit, MPSGraphTensorNamedDataLayoutNCHW and OIHW.
const PADDING_EXPLICIT: u64 = 0;
const LAYOUT_NCHW: u64 = 0;
const LAYOUT_OIHW: u64 = 2;

fn shape(dims: &[usize]) -> ObjcId {
    unsafe {
        let numbers: Vec<ObjcId> = dims.iter().map(|&d| msg_send![class!(NSNumber), numberWithLongLong: d as i64]).collect();
        msg_send![class!(NSArray), arrayWithObjects: numbers.as_ptr() count: numbers.len() as u64]
    }
}

fn half(v: f32) -> u16 {
    // Round to nearest even through f32 bits (normal range only: the probe's values are).
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    let mant = b & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

fn unhalf(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    if exp == 0 {
        return f32::from_bits(sign) + if mant == 0 { 0.0 } else { (mant as f32) * 2f32.powi(-24) };
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[test]
fn a_cutie_convolution_on_mpsgraph_in_fp16() {
    // (name, batch, in, out, k, stride, w, h)
    let cases = [
        ("res2.0.conv2", 1usize, 64usize, 64usize, 3usize, 1usize, 240usize, 136usize),
        ("layer3.0.conv2", 1, 256, 256, 3, 2, 120, 68),
        ("up_8_4.conv1", 2, 128, 128, 3, 1, 240, 136),
        ("stem", 1, 3, 64, 7, 2, 960, 544),
    ];
    unsafe {
        let device = MTLCreateSystemDefaultDevice();
        assert!(!device.is_null());
        let queue: ObjcId = msg_send![device, newCommandQueue];
        for &(name, n, ci, co, k, s, w, h) in &cases {
            let pad = k / 2;
            let (ow, oh) = ((w + 2 * pad - k) / s + 1, (h + 2 * pad - k) / s + 1);
            let mut rng = Rng(0x5eed + ci as u64);
            let x: Vec<f32> = (0..n * ci * w * h).map(|_| rng.next()).collect();
            let scale = 1.0 / ((ci * k * k) as f32).sqrt();
            let wt: Vec<f32> = (0..co * ci * k * k).map(|_| rng.next() * scale).collect();
            let bias: Vec<f32> = (0..co).map(|_| rng.next() * 0.1).collect();

            let graph: ObjcId = msg_send![class!(MPSGraph), new];
            let nil: ObjcId = std::ptr::null_mut();
            let xp: ObjcId = msg_send![graph, placeholderWithShape: shape(&[n, ci, h, w]) dataType: MPS_FLOAT16 name: nil];
            let w16: Vec<u16> = wt.iter().map(|&v| half(v)).collect();
            let wdata: ObjcId = msg_send![class!(NSData), dataWithBytes: w16.as_ptr() as *const c_void length: (w16.len() * 2) as u64];
            let wc: ObjcId = msg_send![graph, constantWithData: wdata shape: shape(&[co, ci, k, k]) dataType: MPS_FLOAT16];
            let b16: Vec<u16> = bias.iter().map(|&v| half(v)).collect();
            let bdata: ObjcId = msg_send![class!(NSData), dataWithBytes: b16.as_ptr() as *const c_void length: (b16.len() * 2) as u64];
            let bc: ObjcId = msg_send![graph, constantWithData: bdata shape: shape(&[1, co, 1, 1]) dataType: MPS_FLOAT16];
            let desc: ObjcId = msg_send![
                class!(MPSGraphConvolution2DOpDescriptor),
                descriptorWithStrideInX: s as u64
                strideInY: s as u64
                dilationRateInX: 1u64
                dilationRateInY: 1u64
                groups: 1u64
                paddingLeft: pad as u64
                paddingRight: pad as u64
                paddingTop: pad as u64
                paddingBottom: pad as u64
                paddingStyle: PADDING_EXPLICIT
                dataLayout: LAYOUT_NCHW
                weightsLayout: LAYOUT_OIHW
            ];
            assert!(!desc.is_null(), "{name}: descriptor");
            let conv: ObjcId = msg_send![graph, convolution2DWithSourceTensor: xp weightsTensor: wc descriptor: desc name: nil];
            let y: ObjcId = msg_send![graph, additionWithPrimaryTensor: conv secondaryTensor: bc name: nil];

            // Our buffers: the input in half, the output in half, shared storage to read back.
            let x16: Vec<u16> = x.iter().map(|&v| half(v)).collect();
            let xbuf: ObjcId = msg_send![device, newBufferWithBytes: x16.as_ptr() as *const c_void length: (x16.len() * 2) as u64 options: 0u64];
            let out_len = n * co * ow * oh;
            let ybuf: ObjcId = msg_send![device, newBufferWithLength: (out_len * 2) as u64 options: 0u64];
            let xd: ObjcId = msg_send![class!(MPSGraphTensorData), alloc];
            let xd: ObjcId = msg_send![xd, initWithMTLBuffer: xbuf shape: shape(&[n, ci, h, w]) dataType: MPS_FLOAT16];
            let yd: ObjcId = msg_send![class!(MPSGraphTensorData), alloc];
            let yd: ObjcId = msg_send![yd, initWithMTLBuffer: ybuf shape: shape(&[n, co, oh, ow]) dataType: MPS_FLOAT16];
            let feeds: ObjcId = msg_send![class!(NSDictionary), dictionaryWithObject: xd forKey: xp];
            let results: ObjcId = msg_send![class!(NSDictionary), dictionaryWithObject: yd forKey: y];

            let run = || {
                let cb: ObjcId = msg_send![queue, commandBuffer];
                let mcb: ObjcId = msg_send![class!(MPSCommandBuffer), commandBufferWithCommandBuffer: cb];
                let _: () = msg_send![graph, encodeToCommandBuffer: mcb feeds: feeds targetOperations: nil resultsDictionary: results executionDescriptor: nil];
                let root: ObjcId = msg_send![mcb, commandBuffer];
                let same = root == cb;
                let _: () = msg_send![mcb, commit];
                let _: () = msg_send![mcb, waitUntilCompleted];
                same
            };
            let same = run();
            let mut times = Vec::new();
            for _ in 0..7 {
                let t = Instant::now();
                run();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(f64::total_cmp);

            // Against f64 on the inputs as half (what the graph saw).
            let contents: *const u16 = msg_send![ybuf, contents];
            let got: Vec<f32> = std::slice::from_raw_parts(contents, out_len).iter().map(|&v| unhalf(v)).collect();
            let xh: Vec<f64> = x16.iter().map(|&v| unhalf(v) as f64).collect();
            let wh: Vec<f64> = w16.iter().map(|&v| unhalf(v) as f64).collect();
            let mut worst = 0f64;
            let mut largest = 0f64;
            // A sample of outputs (every 97th), the full sum for each.
            for i in (0..out_len).step_by(97) {
                let (b, rest) = (i / (co * ow * oh), i % (co * ow * oh));
                let (o, p) = (rest / (ow * oh), rest % (ow * oh));
                let (oy, ox) = (p / ow, p % ow);
                let mut acc = unhalf(b16[o]) as f64;
                for c in 0..ci {
                    for ky in 0..k {
                        for kx in 0..k {
                            let iy = (oy * s + ky) as isize - pad as isize;
                            let ix = (ox * s + kx) as isize - pad as isize;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                acc += xh[((b * ci + c) * h + iy as usize) * w + ix as usize] * wh[((o * ci + c) * k + ky) * k + kx];
                            }
                        }
                    }
                }
                largest = largest.max(acc.abs());
                worst = worst.max((got[i] as f64 - acc).abs());
            }
            println!(
                "mpsgraph {name}: {n}x{ci} {w}x{h} -> {co} k{k} s{s}: median {:.3} ms (min {:.3}), worst rel {:.2e} of the largest, the command buffer kept: {same}",
                times[3], times[0], worst / largest
            );
            assert!(worst / largest < 5e-3, "{name}: rel {:.2e}", worst / largest);
        }
    }
}
