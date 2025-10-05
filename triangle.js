'use strict'

import { mat4, vec3 } from 'https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.js';

const init = async () => {
    // ~~ INITIALIZE ~~ Make sure we can initialize WebGPU
    if (!navigator.gpu) {
        console.error(
        "WebGPU cannot be initialized - navigator.gpu not found"
        );
        return null;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.error("WebGPU cannot be initialized - Adapter not found");
        return null;
    }
    const device = await adapter.requestDevice();
    device.lost.then(() => {
        console.error("WebGPU cannot be initialized - Device has been lost");
        return null;
    });

    const canvas = document.getElementById("canvas-container");
    //make canvas cover the entire window
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const context = canvas.getContext("webgpu");
    if (!context) {
        console.error(
        "WebGPU cannot be initialized - Canvas does not support WebGPU"
        );
        return null;
    }
    // ~~ CONFIGURE THE SWAP CHAIN ~~
    const devicePixelRatio = window.devicePixelRatio || 1;
    const presentationSize = [
        canvas.clientWidth * devicePixelRatio,
        canvas.clientHeight * devicePixelRatio,
    ];
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat(adapter);

    context.configure({
        device,
        format: presentationFormat,
        size: presentationSize,
    });

    //TO DO: later - handle resize
    //Handles window resize
    // const observer = new ResizeObserver(entries => {
    //     for (const entry of entries) {
    //         const canvas = entry.target;
    //         const width = entry.contentBoxSize[0].inlineSize;
    //         const height = entry.contentBoxSize[0].blockSize;
    //         canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
    //         canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
    //     }

    //     const devicePixelRatio = window.devicePixelRatio || 1;
    //     const presentationSize = [
    //         canvas.clientWidth * devicePixelRatio,
    //         canvas.clientHeight * devicePixelRatio,
    //     ];
    //     context.configure({
    //         device,
    //         format: presentationFormat,
    //         size: presentationSize,
    //     });
    // });
    // observer.observe(canvas);

    const multisampleCount = 4;

    // ~~ SETUP VERTICES (position (vec3<f32>), color(vec3<f32>)) ~~
    // Pack them all into one array
    // Each vertex has a position and a color packed in memory in X Y Z R G B order
    const cellWidth = 16;
    const cellHeight = 16;
    const squareSideSize = 4.0;
    const cellSideSizeX = squareSideSize / cellWidth;
    const cellSideSizeZ = squareSideSize / cellHeight;
    //num of floats per vertex (XYZW) * num of vertices per row * num of vertices per column
    let vertices = new Float32Array(4 * (cellWidth + 1) * (cellHeight + 1));
    let offset = 0;
    const addVertex = (x,z) => {
        vertices[offset++] = x;
        vertices[offset++] = 0.0;
        vertices[offset++] = z;
        vertices[offset++] = 1.0;
    };
    for (let w = 0; w < cellWidth + 1; w++) {
        for (let h = 0; h < cellHeight + 1; h++) {
            const x = (w - cellWidth/2) * cellSideSizeX;
            const z = (h - cellHeight/2) * cellSideSizeZ;
            addVertex(x, z);
        }
    }
    const vertexBuffer = device.createBuffer({
        label: "vertex buffer",
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
    vertexBuffer.unmap();

    const vertexBuffersDescriptors = [
        {
        attributes: [
            {
            shaderLocation: 0,
            offset: 0,
            format: "float32x4",
            }
        ],
        arrayStride: 16,
        stepMode: "vertex",
        },
    ];

    // ~~ SETUP INDICES ~~
    let indices = new Int32Array(6 * cellWidth * cellHeight);
    let offsetIdx = 0;
    let col = cellHeight + 1;
    for (let w = 0; w < cellWidth; ++w) {
        for (let h = 0; h < cellHeight; ++h) {
            indices[offsetIdx++] = w * col + h;
            indices[offsetIdx++] = (w + 1) * col + h;
            indices[offsetIdx++] = (w + 1) * col + h + 1;
            indices[offsetIdx++] = w * col + h;
            indices[offsetIdx++] = (w + 1) * col + h + 1;
            indices[offsetIdx++] = w * col + h + 1;
        }
    }

    let indexBuffer = device.createBuffer({
        label: "index buffer",
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Int32Array(indexBuffer.getMappedRange()).set(indices);
    indexBuffer.unmap();

    // ~~ SETUP UNIFORM BUFFER FOR DISPLAY SHADER MODULE ~~
    const objectPosition = [0, 0, 0];
    const cameraPosition = [0, 2, 4];
    const target = [0, 0, 0];
    const up = [0, 1, 0];
    const fovX = 60 * Math.PI / 180;
    const aspect = canvas.clientWidth / canvas.clientHeight;
    const near = 0.1;
    const far = 100;

    const modelMatrix = mat4.translation(objectPosition);
    const viewMatrix = mat4.lookAt(cameraPosition, target, up);
    const projectionMatrix = mat4.perspective(fovX, aspect, near, far);

    const mvpMatrix = mat4.multiply(projectionMatrix, mat4.multiply(viewMatrix, modelMatrix));
    const column = new Uint32Array([cellHeight + 1]);
    const uniformBuffer = device.createBuffer({
        size: mvpMatrix.byteLength + 16,// + column.byteLength doesn't work due to padding and alinment issues
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(uniformBuffer.getMappedRange(0, mvpMatrix.byteLength)).set(mvpMatrix);
    new Uint32Array(uniformBuffer.getMappedRange(mvpMatrix.byteLength, column.byteLength)).set(column);
    uniformBuffer.unmap();

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

    // ~~ DEFINE DISPLAY SHADER ~~
    //TO DO: find a way to load from a separate file
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

    const dispayShaderModule = device.createShaderModule({
        code: displayCode
    });

    // ~~ SETUP UNIFORM BUFFERS FOR COMPUTE SHADER MODULE ~~
    // const computeUniformBuffer1 = createBuffer({
    //     size: 16,
    //     usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    //     mappedAtCreation: true,
    // });
    // new Float32Array(computeUniformBuffer1.getMappedRange(0, 8)).set(new Float32Array([squareSideSize, squareSideSize]));
    // new Uint32Array(computeUniformBuffer1.getMappedRange(0, 8)).set(new Uint32Array([cellWidth + 1, cellHeight + 1]));
    // computeUniformBuffer1.unmap();
    
    const computeUniformBuffer1 = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

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

    // ~~ SETUP COMPUTE SHADER MODULE ~~
    const computeCode = /* wgsl */`
        @group(0) @binding(0) var<storage, read_write> points : array<f32>;
        @group(1) @binding(0) var<uniform> time : f32;

        override size_x: f32 = 4.0;
        override size_z: f32 = 4.0;
        override numCells_x: u32 = 16;
        override numCells_z: u32 = 16;

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
    
    const computeShaderModule = device.createShaderModule({
        code: computeCode
    });

    // ~~ CREATE RENDER PIPELINE ~~
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = device.createRenderPipeline({
        vertex: {
        module: dispayShaderModule,
        entryPoint: "vertex_main",
        buffers: vertexBuffersDescriptors,
        },
        fragment: {
        module: dispayShaderModule,
        entryPoint: "fragment_main",
        targets: [
            {
            format: presentationFormat,
            },
        ],
        },
        primitive: {
        topology: "triangle-list",
        cullmode: "none",
        },
        multisample: {
            count: multisampleCount,
        },
        layout: pipelineLayout,
    });

    // ~~ CREATE COMPUTE PIPELINE ~~
    const computePipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [computeBindGroupLayout0, computeBindGroupLayout1]
    });

    const computePipeline = device.createComputePipeline({
        label: "compute pipeline",
        compute: {
        module: computeShaderModule,
        constants: {
            size_x: squareSideSize, 
            size_z: squareSideSize,
            numCells_x: cellWidth + 1,
            numCells_z: cellHeight + 1,
        },
        entryPoint: "compute",
        },
        layout: computePipelineLayout
    });

    // ~~ CREATE BIND GROUP FOR UNIFORM BUFFER ~~
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
                buffer: computeUniformBuffer1
                }
            },
        ],
    });

    // ~~ CREATE RENDER PASS DESCRIPTOR ~~
    const renderPassDescriptor = {
        colorAttachments: [
        {
            clearValue: { r: 0.0, g: 0.1, b: 0.0, a: 1.0 },
            loadOp: "clear",
            storeOp: "store",
        },
        ],
    };

    let multisampleTexture;
    // ~~ Define render loop ~~
    let timeInit = performance.now();
    function frame() {
        const canvasTexture = context.getCurrentTexture();
        // ~~ CREATE MULTISAMPLE TEXTURE IF IT IS NOT YET CREATED OR HAS WRONG SIZE ~~
        if (!multisampleTexture ||
                multisampleTexture.width !== canvasTexture.width ||
                multisampleTexture.height !== canvasTexture.height) {
        
            // If we have an existing multisample texture destroy it.
            if (multisampleTexture) {
                multisampleTexture.destroy();
            }
        
            // Create a new multisample texture that matches our canvas's size
            multisampleTexture = device.createTexture({
                format: canvasTexture.format,
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
                size: [canvasTexture.width, canvasTexture.height],
                sampleCount: multisampleCount,
            });
        }

        let time = (performance.now() - timeInit) / 10000.0;
        device.queue.writeBuffer(computeUniformBuffer1, 0, new Float32Array([time, 0, 0, 0]), 0, 4);
        renderPassDescriptor.colorAttachments[0].view = multisampleTexture.createView();
        renderPassDescriptor.colorAttachments[0].resolveTarget = canvasTexture.createView();

        const commandEncoder = device.createCommandEncoder();

        const computePassEncoder = commandEncoder.beginComputePass();
        computePassEncoder.setPipeline(computePipeline);
        computePassEncoder.setBindGroup(0, computeBindGroup0);
        computePassEncoder.setBindGroup(1, computeBindGroup1);
        computePassEncoder.dispatchWorkgroups((cellHeight + 1) * (cellWidth + 1));
        computePassEncoder.end();

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

        passEncoder.setPipeline(pipeline);
        passEncoder.setVertexBuffer(0, vertexBuffer);
        passEncoder.setIndexBuffer(indexBuffer, "uint32");
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.drawIndexed(cellHeight * cellWidth * 6);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
};

init();