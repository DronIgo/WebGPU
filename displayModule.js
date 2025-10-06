
const displayCode = /* wgsl */`
struct VertexOut {
    @builtin(position) position : vec4<f32>,
    @location(0) bar : vec2<f32>,
};

struct Uniforms {
    mvp : mat4x4<f32>,
    col : u32,
}

@group(0) @binding(0) var<uniform> uni : Uniforms;

@vertex
fn vertex_main(
    @builtin(vertex_index) vertexIndex : u32,
    @location(0) position: vec4<f32>
) -> VertexOut
{
    var output : VertexOut;
    output.position = uni.mvp * position;
    var xIndex = vertexIndex % uni.col;
    var yIndex = vertexIndex / uni.col;
    //analogue of baricentric coordinates for each cell values will look similiar to following:
    //(0,1)--(1,1)
    //  |      |
    //(0,0)--(1,0)
    //which will allow as to check whether pixel is close to the side of cell or not in the fragment shader
    output.bar = vec2(f32(xIndex % 2), f32(yIndex % 2));
    return output;
} 

//desired width of wireframe lines in pixels
const lineWidth : f32 = 1.0;
@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
{
    var bar = fragData.bar;
    //essentialy speed of changing for each of the "baricentric" coordinates
    var d = fwidth(fragData.bar);
    var bar4 = vec4(bar, vec2(1.0) - bar);
    var thold4 = vec4(d * lineWidth, d * lineWidth);
    var onLine = any(vec4<bool>(bar4 < thold4));
    return select(vec4(0.8, 0.8, 0.8, 1.0), vec4(0.0, 0.0, 0.0, 1.0), onLine);
} `;

import { mat4, vec3 } from 'https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.js';
export function prepareDisplayShaderModule(device, screen_aspect, NUM_CELLS_Z) {
    let result = {};

    // ~~ CREATE SHADER MODULE ~~
    const dispayShaderModule = device.createShaderModule({
        code: displayCode
    });
    result.shaderModule = dispayShaderModule;

     // ~~ SETUP UNIFORM BUFFER FOR DISPLAY SHADER MODULE ~~
    const objectPosition = [0, 0, 0];
    const cameraPosition = [0, 2, 4];
    const target = [0, 0, 0];
    const up = [0, 1, 0];
    const fovX = 60 * Math.PI / 180;
    const aspect = screen_aspect;
    const near = 0.1;
    const far = 100;

    const modelMatrix = mat4.translation(objectPosition);
    const viewMatrix = mat4.lookAt(cameraPosition, target, up);
    const projectionMatrix = mat4.perspective(fovX, aspect, near, far);

    const mvpMatrix = mat4.multiply(projectionMatrix, mat4.multiply(viewMatrix, modelMatrix));
    const column = new Uint32Array([NUM_CELLS_Z + 1]);
    const uniformBuffer = device.createBuffer({
        size: mvpMatrix.byteLength + 16,// + column.byteLength doesn't work due to padding and alinment issues
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(uniformBuffer.getMappedRange(0, mvpMatrix.byteLength)).set(mvpMatrix);
    new Uint32Array(uniformBuffer.getMappedRange(mvpMatrix.byteLength, column.byteLength)).set(column);
    uniformBuffer.unmap();

    result.uniformBuffers = [uniformBuffer];

    // ~~CREATE BIND GROUP LAYOUT ~~
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {
                    type: "uniform",
                },
            }
        ]
    });
    result.bindGroupLayouts = [bindGroupLayout];

    // ~~ CREATE BIND GROUP ~~
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: uniformBuffer 
                }
            },
        ],
    });
    result.bindGroups = [bindGroup];
    return result;
}