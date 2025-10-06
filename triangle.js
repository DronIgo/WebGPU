'use strict'

import {MULTISAMPLE_COUNT, NUM_CELLS_X, NUM_CELLS_Z, CLOTH_SIDE_SIZE} from './constants.js'
import { prepareDisplayShaderModule } from './displayModule.js';
import { prepareComputeShaderModule } from './computeModule.js';

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

    // ~~ SETUP VERTICES (position (vec4<f32>)) ~~
    const cellSideSizeX = CLOTH_SIDE_SIZE / NUM_CELLS_X;
    const cellSideSizeZ = CLOTH_SIDE_SIZE / NUM_CELLS_Z;
    //num of floats per vertex (XYZW) * num of vertices per row * num of vertices per column
    let vertices = new Float32Array(4 * (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1));
    let offset = 0;
    const addVertex = (x,z) => {
        vertices[offset++] = x;
        vertices[offset++] = 0.0;
        vertices[offset++] = z;
        vertices[offset++] = 1.0;
    };
    for (let w = 0; w < NUM_CELLS_X + 1; w++) {
        for (let h = 0; h < NUM_CELLS_Z + 1; h++) {
            const x = (w - NUM_CELLS_X/2) * cellSideSizeX;
            const z = (h - NUM_CELLS_Z/2) * cellSideSizeZ;
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
    let indices = new Int32Array(6 * NUM_CELLS_X * NUM_CELLS_Z);
    let offsetIdx = 0;
    let col = NUM_CELLS_Z + 1;
    for (let w = 0; w < NUM_CELLS_X; ++w) {
        for (let h = 0; h < NUM_CELLS_Z; ++h) {
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
    
    // ~~ PREPARE DISPLAY SHADER MODULE (as well as all related uniform buffers, bind group layouts and bind groups) ~~
    let dispSM = prepareDisplayShaderModule(device, canvas.clientWidth / canvas.clientHeight, NUM_CELLS_Z);

    // ~~ PREPARE COMPUTE SHADER MODULE (as well as all related uniform buffers, bind group layouts and bind groups) ~~
    let computeSM = prepareComputeShaderModule(device, vertexBuffer);

    // ~~ CREATE RENDER PIPELINE ~~
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: dispSM.bindGroupLayouts
    });

    const pipeline = device.createRenderPipeline({
        vertex: {
        module: dispSM.shaderModule,
        entryPoint: "vertex_main",
        buffers: vertexBuffersDescriptors,
        },
        fragment: {
        module: dispSM.shaderModule,
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
            count: MULTISAMPLE_COUNT,
        },
        layout: pipelineLayout,
    });

    // ~~ CREATE COMPUTE PIPELINE ~~
    const computePipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: computeSM.bindGroupLayouts
    });

    const computePipeline = device.createComputePipeline({
        label: "compute pipeline",
        compute: {
        module: computeSM.shaderModule,
        constants: {
            size_x: CLOTH_SIDE_SIZE, 
            size_z: CLOTH_SIDE_SIZE,
            numCells_x: NUM_CELLS_X + 1,
            numCells_z: NUM_CELLS_Z + 1,
        },
        entryPoint: "compute",
        },
        layout: computePipelineLayout
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
    let frames = 0;
    let timePrev = 0.0;
    let timePass = 0.0;
    function frame() {
        frames++;
        let time = (performance.now() - timeInit) / 1000.0;
        let deltaTime = time - timePrev;
        timePrev = time;
        timePass += deltaTime;
        if (timePass > 1.0) {
            timePass -= 1.0;
            console.debug("fps: ", frames);
            frames = 0;
        }
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
                sampleCount: MULTISAMPLE_COUNT,
            });
        }

        device.queue.writeBuffer(computeSM.uniormBuffer, 0, new Float32Array([time, 0, 0, 0]), 0, 4);
        renderPassDescriptor.colorAttachments[0].view = multisampleTexture.createView();
        renderPassDescriptor.colorAttachments[0].resolveTarget = canvasTexture.createView();

        const commandEncoder = device.createCommandEncoder();

        const computePassEncoder = commandEncoder.beginComputePass();
        computePassEncoder.setPipeline(computePipeline);
        computeSM.bindGroups.forEach((bg, idx) => {
            computePassEncoder.setBindGroup(idx, bg);
        });
        computePassEncoder.dispatchWorkgroups((NUM_CELLS_Z + 1) * (NUM_CELLS_X + 1));
        computePassEncoder.end();

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

        passEncoder.setPipeline(pipeline);
        passEncoder.setVertexBuffer(0, vertexBuffer);
        passEncoder.setIndexBuffer(indexBuffer, "uint32");
        dispSM.bindGroups.forEach((bg, idx) => {
            passEncoder.setBindGroup(idx, bg);
        });
        passEncoder.drawIndexed(NUM_CELLS_Z * NUM_CELLS_X * 6);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
};

init();