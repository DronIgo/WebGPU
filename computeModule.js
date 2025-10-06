const computeCode = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> points : array<f32>;
@group(1) @binding(0) var<uniform> time : f32;

override size_x: f32 = 4.0;
override size_z: f32 = 4.0;
override numCells_x: u32 = 4;
override numCells_z: u32 = 4;

@compute @workgroup_size(1) 
fn compute(@builtin(global_invocation_id) id : vec3<u32>)
{
    var xIdx = id.x / numCells_z;
    var zIdx = id.x % numCells_z;
    var x = f32(xIdx) * (size_x / f32(numCells_x)) - (size_x / 2.0);
    var z = f32(zIdx) * (size_z / f32(numCells_z)) - (size_z / 2.0);
    points[4 * id.x + 1] = sin(time + x);
}
`;

export function prepareComputeShaderModule(device, vertexBuffer) {
    let result = {};

    // ~~ CREATE COMPUTE SHADER MODULE ~~
    const computeShaderModule = device.createShaderModule({
        code: computeCode
    });
    result.shaderModule = computeShaderModule;

    // ~~ CREATE COMPUTE UNIFORM BUFFER ~~
    const computeUniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    result.uniormBuffer = computeUniformBuffer;

    // ~~ CREATE BIND GROUPS LAYOUTS ~~
    const computeBindGroupLayout0 = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                },
            }
        ]
    });
    const computeBindGroupLayout1 = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform",
                },
            }
        ]
    });

    result.bindGroupLayouts = [computeBindGroupLayout0, computeBindGroupLayout1];

    // ~~ CREATE BIND GROUP FOR UNIFORM BUFFER ~~
    const computeBindGroup0 = device.createBindGroup({
        layout: computeBindGroupLayout0,
        entries: [
            {
            binding: 0,
            resource:
                {
                buffer: vertexBuffer
                }
            },
        ],
    });

    const computeBindGroup1 = device.createBindGroup({
        layout: computeBindGroupLayout1,
        entries: [
            {
            binding: 0,
            resource: 
                {
                buffer: computeUniformBuffer
                }
            },
        ],
    });

    result.bindGroups = [computeBindGroup0, computeBindGroup1];
    return result;
}