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
    points[3 * id.x + 1] = sin(time + x);
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

//https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
export function projectStretchConstraint(p1, p2, d, w1, w2) {
    let diff = vecSub(p1, p2);
    let diffNorm = vecNorm(diff);
    let diffLen = vecLength(diff);
    let dp1 = vecMulS(diffNorm, (-w1 / (w1 + w2)) * (diffLen - d));
    let dp2 = vecMulS(diffNorm, (w2 / (w1 + w2)) * (diffLen - d));
    for (let i = 0; i < 3; ++i) {
        p1[i] += dp1[i];
        p2[i] += dp2[i];
    }
    return [p1, p2];
}

//https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
export function projectBendConstraint(_p1, _p2, _p3, _p4, phi, w1, w2, w3, w4) {
    let p2 = vecSub(_p2, _p1);
    let p3 = vecSub(_p3, _p1);
    let p4 = vecSub(_p4, _p1);
    let c23 = cross(p2, p3);
    let c24 = cross(p2, p4);
    let n1 = vecNorm(c23);
    let n2 = vecNorm(c24);
    let d = vecMul(n1, n2);

    let q3 = vecDiv(vecAdd(cross(p2, n2), vecMulS(cross(n1, p2), d)), vecLength(c23));
    let q4 = vecDiv(vecAdd(cross(p2, n1), vecMulS(cross(n2, p2), d)), vecLength(c24));
    let q2 = vecSub([0,0,0], vecAdd(vecDiv(vecAdd(cross(p3, n2), vecMulS(cross(n1, p3), d)), vecLength(c23)),
        vecDiv(vecAdd(cross(p4, n1), vecMulS(cross(n2, p4), d)), vecLength(c24))));
    let q1 = vecSub(vecSub(vecSub([0,0,0], q2), q3), q4);

    let denom = w1 * vecMul(q1, q1) + w2 * vecMul(q2, q2) + 
        w3 * vecMul(q3, q3) + w4 * vecMul(q4, q4);
    
    if (denom < 1e-12) {
        return [_p1, _p2, _p3, _p4];
    }

    let common = Math.sqrt(1 - d * d) * (Math.acos(d) - phi) / denom;
    let dp1 = vecMulS(q1, -w1 * common);
    let dp2 = vecMulS(q2, -w2 * common);
    let dp3 = vecMulS(q3, -w3 * common);
    let dp4 = vecMulS(q4, -w4 * common);
    return [vecAdd(_p1,dp1), vecAdd(_p2, dp2), vecAdd(_p3, dp3), vecAdd(_p4, dp4)];
}

function cross(a, b) {
    return [ (a[1] * b[2] - a[2] * b[1]), (a[2] * b[0] - a[0] * b[2]), (a[0] * b[1] - a[1] * b[0]) ];
}

function vecMul(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vecLength(a) {
    return Math.sqrt(vecMul(a, a));
}

function vecSub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vecAdd(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vecDiv(a, s) {
    return [a[0] / s, a[1] / s, a[2] / s];
}

function vecMulS(a, s) {
    return [a[0] * s, a[1] * s, a[2] * s];
}

function vecNorm(a) {
    return vecDiv(a, vecLength(a));
}