// SHADER FOR PROJECTING STRETCH CONSTRAINTS
const computeStretchCode = /* wgsl */`
struct StretchConstrIdx {
    idx1 : u32,
    idx2 : u32,
};

struct DeltasStretch {
    dp1 : vec3<f32>,
    dp2 : vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> vertices : array<f32>;
//group 1 is not updated in runtime
@group(1) @binding(0) var<storage, read> invMass : array<f32>;

@group(2) @binding(0) var<storage, read> stretchIdx : array<StretchConstrIdx>;
@group(2) @binding(1) var<storage, read> stretchD : array<f32>;
@group(2) @binding(2) var<uniform> numConstr : u32;

override size_x: f32 = 4.0;
override size_z: f32 = 4.0;
override numCells_x: u32 = 4;
override numCells_z: u32 = 4;

override wg_x: i32 = 64;
@compute @workgroup_size(wg_x, 1, 1) 
fn compute(@builtin(global_invocation_id) id : vec3<u32>)
{   
    if (id.x >= numConstr) {
        return;
    }
    var idx1 = stretchIdx[id.x].idx1;
    var idx2 = stretchIdx[id.x].idx2;
    var d = stretchD[id.x];
    var p1 = vec3<f32>(vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]);
    var p2 = vec3<f32>(vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]);
    var w1 = invMass[idx1];
    var w2 = invMass[idx2];
    var deltas = projectStretch(p1, p2, d, w1, w2);
    vertices[idx1 * 3] = p1.x + deltas.dp1.x;
    vertices[idx1 * 3 + 1] = p1.y + deltas.dp1.y;
    vertices[idx1 * 3 + 2] = p1.z + deltas.dp1.z;

    vertices[idx2 * 3] = p2.x + deltas.dp2.x;
    vertices[idx2 * 3 + 1] = p2.y + deltas.dp2.y;
    vertices[idx2 * 3 + 2] = p2.z + deltas.dp2.z;
}

fn projectStretch(p1 : vec3<f32>, p2 : vec3<f32>, d : f32, w1 : f32, w2 : f32) -> DeltasStretch{
    var diff = p1 - p2;
    var diffNorm = normalize(diff);
    var diffLen = length(diff);
    var denom = w1 + w2;
    var dp1 = select(vec3<f32>(0.0, 0.0, 0.0), diffNorm * (-w1 / denom) * (diffLen - d), denom != 0.0);
    var dp2 = select(vec3<f32>(0.0, 0.0, 0.0), diffNorm * (w2 / denom) * (diffLen - d), denom != 0.0);
    var deltas : DeltasStretch;
    deltas.dp1 = dp1;
    deltas.dp2 = dp2;
    return deltas;
}
`;

// SHADER FOR PROJECTING BEND CONSTRAINTS
const computeBendCode = /* wgsl */`
struct BendConstrIdx {
    idx1 : u32,
    idx2 : u32,
    idx3 : u32,
    idx4 : u32,
};

@group(0) @binding(0) var<storage, read_write> vertices : array<f32>;
//group 1 is not updated in runtime
@group(1) @binding(0) var<storage, read> invMass : array<f32>;

@group(2) @binding(0) var<storage, read> bendIdx : array<BendConstrIdx>;
@group(2) @binding(1) var<storage, read> bendPhi : array<f32>;
@group(2) @binding(2) var<uniform> numConstr : u32;

override size_x: f32 = 4.0;
override size_z: f32 = 4.0;
override numCells_x: u32 = 4;
override numCells_z: u32 = 4;

override wg_x: i32 = 64;
@compute @workgroup_size(wg_x, 1, 1) 
fn compute(@builtin(global_invocation_id) id : vec3<u32>)
{   
    if (id.x >= numConstr) {
        return;
    }
    var idx1 = bendIdx[id.x].idx1;
    var idx2 = bendIdx[id.x].idx2;
    var idx3 = bendIdx[id.x].idx3;
    var idx4 = bendIdx[id.x].idx4;
    var phi = bendPhi[id.x];
    var p1 = vec3<f32>(vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]);
    var p2 = vec3<f32>(vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]);
    var p3 = vec3<f32>(vertices[idx3 * 3], vertices[idx3 * 3 + 1], vertices[idx3 * 3 + 2]);
    var p4 = vec3<f32>(vertices[idx4 * 3], vertices[idx4 * 3 + 1], vertices[idx4 * 3 + 2]);
    var w1 = invMass[idx1];
    var w2 = invMass[idx2];
    var w3 = invMass[idx3];
    var w4 = invMass[idx4];
    var deltas = projectBend(p1, p2, p3, p4, phi, w1, w2, w3, w4);
    vertices[idx1 * 3] = p1.x + deltas.dp1.x;
    vertices[idx1 * 3 + 1] = p1.y + deltas.dp1.y;
    vertices[idx1 * 3 + 2] = p1.z + deltas.dp1.z;

    vertices[idx2 * 3] = p2.x + deltas.dp2.x;
    vertices[idx2 * 3 + 1] = p2.y + deltas.dp2.y;
    vertices[idx2 * 3 + 2] = p2.z + deltas.dp2.z;

    vertices[idx3 * 3] = p3.x + deltas.dp3.x;
    vertices[idx3 * 3 + 1] = p3.y + deltas.dp3.y;
    vertices[idx3 * 3 + 2] = p3.z + deltas.dp3.z;

    vertices[idx4 * 3] = p4.x + deltas.dp4.x;
    vertices[idx4 * 3 + 1] = p4.y + deltas.dp4.y;
    vertices[idx4 * 3 + 2] = p4.z + deltas.dp4.z;
}

struct DeltasBend{
    dp1 : vec3<f32>,
    dp2 : vec3<f32>,
    dp3 : vec3<f32>,
    dp4 : vec3<f32>,
};

fn projectBend(_p1 : vec3<f32>, _p2 : vec3<f32>, _p3 : vec3<f32>, _p4 : vec3<f32>, phi : f32, 
                w1 : f32, w2 : f32, w3 : f32, w4 : f32) -> DeltasBend {
    var p2 = _p2 - _p1;
    var p3 = _p3 - _p1;
    var p4 = _p4 - _p1;
    var s1 = select(0.0, 1.0, w1 > 0.0);
    var s2 = select(0.0, 1.0, w2 > 0.0);
    var s3 = select(0.0, 1.0, w3 > 0.0);
    var s4 = select(0.0, 1.0, w4 > 0.0);
    var c23 = cross(p2, p3);
    var c24 = cross(p2, p4);
    var n1 = normalize(c23);
    var n2 = normalize(c24);
    var d = dot(n1, n2);
    d = clamp(d, -1.0, 1.0);
    var q3 = s3 * (cross(p2, n2) + (cross(n1, p2) * d)) / length(c23);
    var q4 = s4 * (cross(p2, n1) + (cross(n2, p2) * d)) / length(c24);
    var q2 = -s2 * ((cross(p3, n2) + (cross(n1, p3) * d)) / length(c23) + 
        (cross(p4, n1) + (cross(n2, p4) * d)) / length(c24));
    var  q1 = -s1 * (q2 + q3 + q4);

    var denom = w1 * dot(q1, q1) + w2 * dot(q2, q2) + 
        w3 * dot(q3, q3) + w4 * dot(q4, q4);

    var com = select(sqrt(1 - d * d) * (acos(d) - phi) / denom, 0.0, denom < 1e-12);
    var dp1 = q1 * (-w1 * com);
    var dp2 = q2 * (-w2 * com);
    var dp3 = q3 * (-w3 * com);
    var dp4 = q4 * (-w4 * com);
    var deltas : DeltasBend;
    deltas.dp1 = dp1;
    deltas.dp2 = dp2;
    deltas.dp3 = dp3;
    deltas.dp4 = dp4;
    return deltas;
}
`;

// SHADER FOR UPDATING SPEED AND POSITION
const computeUpdatePositionCode = /* wgsl */`

struct PerFrameVars
{
    deltaTime: f32,
    gravity: f32,
    time: f32,
};

@group(0) @binding(0) var<storage, read> verticesR : array<f32>;
@group(0) @binding(1) var<storage, read_write> verticesW : array<f32>;

@group(1) @binding(0) var<storage, read> invMass : array<f32>;

@group(2) @binding(0) var<storage, read> velocities : array<f32>;

@group(3) @binding(0) var<uniform> perFrame : PerFrameVars;

override size_x: f32 = 4.0;
override size_z: f32 = 4.0;
override numCells_x: u32 = 4;
override numCells_z: u32 = 4;
override idxOfCenter: u32 = 12;
override amp: f32 = 0.5;
override wg_x: i32 = 64;
@compute @workgroup_size(wg_x, 1, 1) 
fn compute(@builtin(global_invocation_id) id : vec3<u32>)
{   
    if (id.x >= (numCells_x + 1) * (numCells_z + 1)) {
        return;
    }
    var s = select(0.0, 1.0, invMass[id.x] > 0.0);
    var v = vec3<f32>(velocities[3 * id.x], velocities[3 * id.x + 1], velocities[3 * id.x + 2]);
    v.y -= perFrame.gravity * perFrame.deltaTime * s;
    if (id.x == idxOfCenter) {
        verticesW[3 * id.x + 1] = sin(perFrame.time) * amp;
    } else {
        verticesW[3 * id.x] = verticesR[3 * id.x] + v.x * perFrame.deltaTime * s;
        verticesW[3 * id.x + 1] = verticesR[3 * id.x + 1] + v.y * perFrame.deltaTime * s;
        verticesW[3 * id.x + 2] = verticesR[3 * id.x + 2] + v.z * perFrame.deltaTime * s;
    }
}
`;

// SHADER FOR UPDATING SPEED AT THE END
const computeUpdateVelocitiesCode = /* wgsl */`

struct PerFrameVars
{
    deltaTime: f32,
    gravity: f32,
};

@group(0) @binding(0) var<storage, read> verticesPrev : array<f32>;
@group(0) @binding(1) var<storage, read> verticesCur : array<f32>;

@group(1) @binding(0) var<storage, read_write> velocities : array<f32>;

@group(2) @binding(0) var<uniform> perFrame : PerFrameVars;

override size_x: f32 = 4.0;
override size_z: f32 = 4.0;
override numCells_x: u32 = 4;
override numCells_z: u32 = 4;

override wg_x: i32 = 64;
@compute @workgroup_size(wg_x, 1, 1) 
fn compute(@builtin(global_invocation_id) id : vec3<u32>)
{   
    if (id.x >= (numCells_x + 1) * (numCells_z + 1)) {
        return;
    }
    velocities[3 * id.x] = (verticesCur[3 * id.x] - verticesPrev[3 * id.x]) / perFrame.deltaTime;
    velocities[3 * id.x + 1] = (verticesCur[3 * id.x + 1] - verticesPrev[3 * id.x + 1]) / perFrame.deltaTime;
    velocities[3 * id.x + 2] = (verticesCur[3 * id.x + 2] - verticesPrev[3 * id.x + 2]) / perFrame.deltaTime;
}
`;

export function prepareComputeShaderModule(device, CLOTH_SIDE_SIZE, NUM_CELLS_X, NUM_CELLS_Z, WORKGROUP_SIZE, SIN_AMP) {
    let result = {};

    // ~~ CREATE COMPUTE SHADER MODULE ~~
    const shaderModuleStretch = device.createShaderModule({
        code: computeStretchCode
    });

    const shaderModuleBend = device.createShaderModule({
        code: computeBendCode
    });

    const shaderModuleUpdateInit = device.createShaderModule({
        code: computeUpdatePositionCode
    });

    const shaderModuleUpdateFinal = device.createShaderModule({
        code: computeUpdateVelocitiesCode
    });

    // ~~ CREATE COMMON UNIFORM BUFFER ~~
    const perFrameUniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    result.perFrameUniormBuffer = perFrameUniformBuffer;

    const numConstrUniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    result.numConstrUniformBuffer = numConstrUniformBuffer;

    // ~~ CREATE COMMON BIND GROUPS LAYOUTS ~~
    const bindGroupLayoutS = device.createBindGroupLayout({
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
    result.bindGroupLayoutS = bindGroupLayoutS;

    const bindGroupLayoutRS = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            }
        ]
    });
    result.bindGroupLayoutRS = bindGroupLayoutRS;

    const bindGroupLayoutRSRS = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            }
        ]
    });
    result.bindGroupLayoutRSRS = bindGroupLayoutRSRS;

    const bindGroupLayoutRSS = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage",
                },
            }
        ]
    });
    result.bindGroupLayoutRSS = bindGroupLayoutRSS;

    const bindGroupLayoutU = device.createBindGroupLayout({
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
    result.bindGroupLayoutU = bindGroupLayoutU;

    const bindGroupLayoutRSRSU = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage",
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "uniform",
                },
            },
        ]
    });
    result.bindGroupLayoutRSRSU = bindGroupLayoutRSRSU;

    // ~~ CREATE COMMON BIND GROUPS ~~
    const bindGroupPerFrame = device.createBindGroup({
        layout: bindGroupLayoutU,
        entries: [
            {
                binding: 0,
                resource:
                {
                    buffer: perFrameUniformBuffer
                }
            },
        ],
    });
    result.bindGroupPerFrame = bindGroupPerFrame;

    // ~~ CREATE STRETCH PIPELINE ~~
    const stretchPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [
            bindGroupLayoutS,
            bindGroupLayoutRS,
            bindGroupLayoutRSRSU
        ]
    });

    const stretchPipeline = device.createComputePipeline({
        label: "stretch pipeline",
        compute: {
            module: shaderModuleStretch,
            constants: {
                size_x: CLOTH_SIDE_SIZE,
                size_z: CLOTH_SIDE_SIZE,
                numCells_x: NUM_CELLS_X + 1,
                numCells_z: NUM_CELLS_Z + 1,
                wg_x: WORKGROUP_SIZE,
            },
            entryPoint: "compute",
        },
        layout: stretchPipelineLayout
    });

    result.stretchPipeline = stretchPipeline;

    // ~~ CREATE BEND PIPELINE ~~
    const bendPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [
            bindGroupLayoutS,
            bindGroupLayoutRS,
            bindGroupLayoutRSRSU
        ]
    });

    const bendPipeline = device.createComputePipeline({
        label: "bind pipeline",
        compute: {
            module: shaderModuleBend,
            constants: {
                size_x: CLOTH_SIDE_SIZE,
                size_z: CLOTH_SIDE_SIZE,
                numCells_x: NUM_CELLS_X + 1,
                numCells_z: NUM_CELLS_Z + 1,
                wg_x: WORKGROUP_SIZE,
            },
            entryPoint: "compute",
        },
        layout: bendPipelineLayout
    });

    result.bendPipeline = bendPipeline;

    // ~~ CREATE UPDATE INIT PIPELINE ~~
    const updateInitPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [
            bindGroupLayoutRSS,
            bindGroupLayoutRS,
            bindGroupLayoutRS,
            bindGroupLayoutU
        ]
    });

    const updateInitPipeline = device.createComputePipeline({
        label: "update init pipeline",
        compute: {
            module: shaderModuleUpdateInit,
            constants: {
                size_x: CLOTH_SIDE_SIZE,
                size_z: CLOTH_SIDE_SIZE,
                numCells_x: NUM_CELLS_X,
                numCells_z: NUM_CELLS_Z,
                wg_x: WORKGROUP_SIZE,
                idxOfCenter: (NUM_CELLS_Z + 1) * (NUM_CELLS_X / 2) + NUM_CELLS_Z / 2,
                amp: SIN_AMP,
            },
            entryPoint: "compute",
        },
        layout: updateInitPipelineLayout
    });

    result.updateInitPipeline = updateInitPipeline;

    // ~~ CREATE UPDATE FINAL PIPELINE ~~
    const updateFinalPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [
            bindGroupLayoutRSRS,
            bindGroupLayoutS,
            bindGroupLayoutU,
        ]
    });

    const updateFinalPipeline = device.createComputePipeline({
        label: "update final pipeline",
        compute: {
            module: shaderModuleUpdateFinal,
            constants: {
                size_x: CLOTH_SIDE_SIZE,
                size_z: CLOTH_SIDE_SIZE,
                numCells_x: NUM_CELLS_X + 1,
                numCells_z: NUM_CELLS_Z + 1,
                wg_x: WORKGROUP_SIZE,
            },
            entryPoint: "compute",
        },
        layout: updateFinalPipelineLayout
    });

    result.updateFinalPipeline = updateFinalPipeline;


    return result;
}

//LEGACY CODE - used before the compute shaders
//https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
//s1 and s2 specify whether points 1 and 2 are simulated
export function projectStretchConstraint(p1, p2, d, w1, w2, s1, s2) {
    if (s1 == 0 && s2 == 0)
        return [p1, p2];
    let diff = vecSub(p1, p2);
    let diffNorm = vecNorm(diff);
    let diffLen = vecLength(diff);
    let denom = s1 * w1 + s2 * w2;
    let dp1 = vecMulS(diffNorm, (-w1 * s1 / denom) * (diffLen - d));
    let dp2 = vecMulS(diffNorm, (w2 * s2 / denom) * (diffLen - d));
    for (let i = 0; i < 3; ++i) {
        p1[i] += dp1[i];
        p2[i] += dp2[i];
    }
    return [p1, p2];
}

//https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
export function projectBendConstraint(_p1, _p2, _p3, _p4, phi, w1, w2, w3, w4, s1, s2, s3, s4) {
    let p2 = vecSub(_p2, _p1);
    let p3 = vecSub(_p3, _p1);
    let p4 = vecSub(_p4, _p1);
    let c23 = cross(p2, p3);
    let c24 = cross(p2, p4);
    let n1 = vecNorm(c23);
    let n2 = vecNorm(c24);
    let d = vecMul(n1, n2);

    let q3 = vecMulS(vecDiv(vecAdd(cross(p2, n2), vecMulS(cross(n1, p2), d)), vecLength(c23)), s3);
    let q4 = vecMulS(vecDiv(vecAdd(cross(p2, n1), vecMulS(cross(n2, p2), d)), vecLength(c24)), s4);
    let q2 = vecMulS(vecSub([0, 0, 0], vecAdd(vecDiv(vecAdd(cross(p3, n2), vecMulS(cross(n1, p3), d)), vecLength(c23)),
        vecDiv(vecAdd(cross(p4, n1), vecMulS(cross(n2, p4), d)), vecLength(c24)))), s2);
    let q1 = vecMulS(vecSub(vecSub(vecSub([0, 0, 0], q2), q3), q4), s1);

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
    return [vecAdd(_p1, dp1), vecAdd(_p2, dp2), vecAdd(_p3, dp3), vecAdd(_p4, dp4)];
}

function cross(a, b) {
    return [(a[1] * b[2] - a[2] * b[1]), (a[2] * b[0] - a[0] * b[2]), (a[0] * b[1] - a[1] * b[0])];
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