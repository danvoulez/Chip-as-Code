//! GPU evaluation backend using wgpu/WGSL compute shaders.
//! This module is only compiled when the `gpu` feature is enabled.
#![cfg(feature = "gpu")]

use crate::chip_ir::{Chip, GateOp, Ref};
use crate::evolve::{DatasetSplit, Sample};
use anyhow::{anyhow, Result};
use blake3::hash;
use bytemuck::{Pod, Zeroable};
use logline::json_atomic;
use std::borrow::Cow;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

pub const MAX_GATES: usize = 256;
pub const WGPU_VERSION: &str = "0.19";
const WORKGROUP_SIZE: u32 = 64;
const REF_GATE_MASK: u32 = 0x8000_0000;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq, Eq)]
pub struct BitPackedSample {
    pub features_lo: u32,
    pub features_hi: u32,
    pub label: u32,
    pub _pad: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitDatasetSplit {
    pub features: usize,
    pub train: Vec<BitPackedSample>,
    pub test: Vec<BitPackedSample>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GpuGate {
    op: u32,
    k: u32,
    inputs_offset: u32,
    inputs_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GpuChip {
    gate_offset: u32,
    gate_len: u32,
    ref_offset: u32,
    ref_len: u32,
    output_ref: u32,
    features: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Uniforms {
    samples: u32,
    features: u32,
    max_gates: u32,
    chip_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[derive(Debug, Clone)]
pub struct GpuEvalMetadata {
    pub adapter: String,
    pub backend: String,
    pub shader_hash: String,
}

pub struct GpuEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
    dataset_train: SampleBuffer,
    dataset_test: SampleBuffer,
    pub metadata: GpuEvalMetadata,
}

struct SampleBuffer {
    buffer: wgpu::Buffer,
    len: usize,
    features: usize,
}

struct PackedChips {
    gates: Vec<GpuGate>,
    refs: Vec<u32>,
    chips: Vec<GpuChip>,
}

pub fn pack_bits(features: &[bool]) -> Result<(u32, u32)> {
    if features.len() > 64 {
        return Err(anyhow!("feature length exceeds 64"));
    }
    let mut lo = 0u32;
    let mut hi = 0u32;
    for (idx, bit) in features.iter().copied().enumerate() {
        if bit {
            if idx < 32 {
                lo |= 1 << idx;
            } else {
                hi |= 1 << (idx - 32);
            }
        }
    }
    Ok((lo, hi))
}

pub fn unpack_bits(lo: u32, hi: u32, len: usize) -> Vec<bool> {
    (0..len)
        .map(|i| {
            if i < 32 {
                (lo >> i) & 1 == 1
            } else {
                (hi >> (i - 32)) & 1 == 1
            }
        })
        .collect()
}

pub fn bitpack_split(split: &DatasetSplit) -> Result<BitDatasetSplit> {
    let features = split.features;
    if features > 64 {
        return Err(anyhow!("gpu backend supports up to 64 features"));
    }
    let pack_sample = |s: &Sample| -> Result<BitPackedSample> {
        let (lo, hi) = pack_bits(&s.x)?;
        Ok(BitPackedSample {
            features_lo: lo,
            features_hi: hi,
            label: if s.y { 1 } else { 0 },
            _pad: 0,
        })
    };
    let train = split.train.iter().map(pack_sample).collect::<Result<Vec<_>>>()?;
    let test = split.test.iter().map(pack_sample).collect::<Result<Vec<_>>>()?;
    Ok(BitDatasetSplit {
        features,
        train,
        test,
    })
}

pub fn shader_hash() -> Result<String> {
    let canon = json_atomic::canonize(&SHADER_WGSL)?;
    Ok(hex::encode(hash(&canon).as_bytes()))
}

pub fn init_gpu(split: &DatasetSplit) -> Result<GpuEvaluator> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow!("no GPU adapter available"))?;
    let adapter_info = adapter.get_info();
    let backend = format!("{:?}", adapter_info.backend);
    let device_desc = wgpu::DeviceDescriptor {
        label: Some("chip-gpu-device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&device_desc, None))?;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("chip-gpu-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_WGSL)),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("chip-gpu-bind-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("chip-gpu-pipeline-layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("chip-gpu-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
    });

    let packed = bitpack_split(split)?;
    let dataset_train = SampleBuffer::from_samples(&device, &packed.train, "train", packed.features);
    let dataset_test = SampleBuffer::from_samples(&device, &packed.test, "test", packed.features);
    let shader_hash = shader_hash()?;

    Ok(GpuEvaluator {
        device,
        queue,
        pipeline,
        bind_layout,
        dataset_train,
        dataset_test,
        metadata: GpuEvalMetadata {
            adapter: adapter_info.name,
            backend,
            shader_hash,
        },
    })
}

impl SampleBuffer {
    fn from_samples(
        device: &wgpu::Device,
        samples: &[BitPackedSample],
        label: &str,
        features: usize,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("chip-dataset-{}", label)),
            contents: bytemuck::cast_slice(samples),
            usage: wgpu::BufferUsages::STORAGE,
        });
        SampleBuffer {
            buffer,
            len: samples.len(),
            features,
        }
    }
}

impl GpuEvaluator {
    pub fn evaluate(&self, chips: &[Chip]) -> Result<(Vec<u32>, Vec<u32>)> {
        let packed = pack_chips(chips)?;
        let train = self.run_once(&packed, &self.dataset_train)?;
        let test = self.run_once(&packed, &self.dataset_test)?;
        Ok((train, test))
    }

    fn run_once(&self, packed: &PackedChips, dataset: &SampleBuffer) -> Result<Vec<u32>> {
        if packed.chips.is_empty() {
            return Ok(Vec::new());
        }
        if packed
            .chips
            .first()
            .map(|c| c.features as usize)
            .unwrap_or(dataset.features)
            != dataset.features
        {
            return Err(anyhow!(
                "chip feature count {} mismatches dataset {}",
                packed
                    .chips
                    .first()
                    .map(|c| c.features as usize)
                    .unwrap_or(0),
                dataset.features
            ));
        }
        let mut gate_fallback = Vec::new();
        let gate_slice: &[GpuGate] = if packed.gates.is_empty() {
            gate_fallback.push(GpuGate {
                op: 0,
                k: 0,
                inputs_offset: 0,
                inputs_len: 0,
            });
            &gate_fallback
        } else {
            &packed.gates
        };
        let gate_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chip-gpu-gates"),
            contents: bytemuck::cast_slice(gate_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut ref_fallback = Vec::new();
        let ref_slice: &[u32] = if packed.refs.is_empty() {
            ref_fallback.push(0u32);
            &ref_fallback
        } else {
            &packed.refs
        };
        let ref_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chip-gpu-refs"),
            contents: bytemuck::cast_slice(ref_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let chip_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chip-gpu-chips"),
            contents: bytemuck::cast_slice(&packed.chips),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size = (packed.chips.len() * std::mem::size_of::<u32>()) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chip-gpu-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chip-gpu-output-staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniforms = Uniforms {
            samples: dataset.len as u32,
            features: packed.chips.first().map(|c| c.features).unwrap_or(0),
            max_gates: MAX_GATES as u32,
            chip_count: packed.chips.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chip-gpu-uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chip-gpu-bind-group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dataset.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ref_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: chip_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("chip-gpu-encoder") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chip-gpu-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (packed.chips.len() as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, output_size);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let slice = staging_buf.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| anyhow!("map_async callback dropped"))??;
        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();
        Ok(result)
    }
}

fn encode_ref(r: &Ref) -> u32 {
    match r {
        Ref::Feature(idx) => *idx as u32,
        Ref::Gate(idx) => (*idx as u32) | REF_GATE_MASK,
    }
}

fn pack_chips(chips: &[Chip]) -> Result<PackedChips> {
    if chips.is_empty() {
        return Ok(PackedChips {
            gates: Vec::new(),
            refs: Vec::new(),
            chips: Vec::new(),
        });
    }
    let features = chips[0].features;
    if features > 64 {
        return Err(anyhow!("gpu backend supports up to 64 features"));
    }
    let mut gates_out = Vec::new();
    let mut refs_out = Vec::new();
    let mut chips_out = Vec::new();

    for chip in chips {
        if chip.features != features {
            return Err(anyhow!("mixed feature counts not supported for gpu eval"));
        }
        if chip.gates.len() > MAX_GATES {
            return Err(anyhow!("chip gates {} exceed max {}", chip.gates.len(), MAX_GATES));
        }
        let gate_offset = gates_out.len() as u32;
        let ref_offset = refs_out.len() as u32;
        for gate in &chip.gates {
            match &gate.op {
                GateOp::And(inputs) => {
                    let start = refs_out.len() as u32;
                    for r in inputs {
                        refs_out.push(encode_ref(r));
                    }
                    gates_out.push(GpuGate {
                        op: 0,
                        k: 0,
                        inputs_offset: start,
                        inputs_len: inputs.len() as u32,
                    });
                }
                GateOp::Or(inputs) => {
                    let start = refs_out.len() as u32;
                    for r in inputs {
                        refs_out.push(encode_ref(r));
                    }
                    gates_out.push(GpuGate {
                        op: 1,
                        k: 0,
                        inputs_offset: start,
                        inputs_len: inputs.len() as u32,
                    });
                }
                GateOp::Not(input) => {
                    let start = refs_out.len() as u32;
                    refs_out.push(encode_ref(input));
                    gates_out.push(GpuGate {
                        op: 2,
                        k: 0,
                        inputs_offset: start,
                        inputs_len: 1,
                    });
                }
                GateOp::Threshold { k, inputs } => {
                    let start = refs_out.len() as u32;
                    for r in inputs {
                        refs_out.push(encode_ref(r));
                    }
                    gates_out.push(GpuGate {
                        op: 3,
                        k: *k as u32,
                        inputs_offset: start,
                        inputs_len: inputs.len() as u32,
                    });
                }
            }
        }
        let output_ref = encode_ref(&chip.output);
        chips_out.push(GpuChip {
            gate_offset,
            gate_len: chip.gates.len() as u32,
            ref_offset,
            ref_len: (refs_out.len() as u32).saturating_sub(ref_offset),
            output_ref,
            features: features as u32,
            _pad: 0,
        });
    }

    Ok(PackedChips {
        gates: gates_out,
        refs: refs_out,
        chips: chips_out,
    })
}

const SHADER_WGSL: &str = r#"
struct Sample {
    features_lo: u32,
    features_hi: u32,
    label: u32,
    _pad: u32,
};

struct Gate {
    op: u32,
    k: u32,
    inputs_offset: u32,
    inputs_len: u32,
};

struct Chip {
    gate_offset: u32,
    gate_len: u32,
    ref_offset: u32,
    ref_len: u32,
    output_ref: u32,
    features: u32,
    _pad: u32,
};

struct Uniforms {
    samples: u32,
    features: u32,
    max_gates: u32,
    chip_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> samples: array<Sample>;
@group(0) @binding(1) var<storage, read> gates: array<Gate>;
@group(0) @binding(2) var<storage, read> refs: array<u32>;
@group(0) @binding(3) var<storage, read> chips: array<Chip>;
@group(0) @binding(4) var<storage, read_write> outputs: array<u32>;
@group(0) @binding(5) var<uniform> uni: Uniforms;

const OP_AND: u32 = 0u;
const OP_OR: u32 = 1u;
const OP_NOT: u32 = 2u;
const OP_THRESH: u32 = 3u;
const REF_GATE_MASK: u32 = 0x80000000u;
const MAX_GATES: u32 = 256u;

fn feature_bit(sample: Sample, idx: u32) -> u32 {
    if idx < 32u {
        return (sample.features_lo >> idx) & 1u;
    }
    let offset = idx - 32u;
    return (sample.features_hi >> offset) & 1u;
}

fn resolve_ref(r: u32, sample: Sample, gates_cache: ptr<function, array<u32, MAX_GATES>>) -> u32 {
    if (r & REF_GATE_MASK) != 0u {
        let gid = r & 0x7fffffffu;
        return (*gates_cache)[gid];
    }
    return feature_bit(sample, r);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chip_idx = global_id.x;
    if chip_idx >= uni.chip_count {
        return;
    }
    let chip = chips[chip_idx];
    var gate_vals: array<u32, MAX_GATES>;
    var correct: u32 = 0u;

    for (var s: u32 = 0u; s < uni.samples; s = s + 1u) {
        let sample = samples[s];
        for (var g: u32 = 0u; g < chip.gate_len; g = g + 1u) {
            let gate = gates[chip.gate_offset + g];
            var v: u32 = 0u;
            switch gate.op {
                case OP_AND: {
                    v = 1u;
                    for (var i: u32 = 0u; i < gate.inputs_len; i = i + 1u) {
                        let ref_idx = refs[gate.inputs_offset + i];
                        if resolve_ref(ref_idx, sample, &gate_vals) == 0u {
                            v = 0u;
                            break;
                        }
                    }
                }
                case OP_OR: {
                    v = 0u;
                    for (var i: u32 = 0u; i < gate.inputs_len; i = i + 1u) {
                        let ref_idx = refs[gate.inputs_offset + i];
                        if resolve_ref(ref_idx, sample, &gate_vals) != 0u {
                            v = 1u;
                            break;
                        }
                    }
                }
                case OP_NOT: {
                    let ref_idx = refs[gate.inputs_offset];
                    v = 1u - resolve_ref(ref_idx, sample, &gate_vals);
                }
                case OP_THRESH: {
                    var count: u32 = 0u;
                    for (var i: u32 = 0u; i < gate.inputs_len; i = i + 1u) {
                        let ref_idx = refs[gate.inputs_offset + i];
                        count = count + resolve_ref(ref_idx, sample, &gate_vals);
                    }
                    if count >= gate.k {
                        v = 1u;
                    } else {
                        v = 0u;
                    }
                }
                default: {
                    v = 0u;
                }
            }
            gate_vals[g] = v;
        }
        let prediction = resolve_ref(chip.output_ref, sample, &gate_vals);
        if prediction == sample.label {
            correct = correct + 1u;
        }
    }

    outputs[chip_idx] = correct;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let bits = vec![true, false, true, false, true, false, false, true];
        let (lo, hi) = pack_bits(&bits).unwrap();
        let round = unpack_bits(lo, hi, bits.len());
        assert_eq!(bits, round);
    }
}
