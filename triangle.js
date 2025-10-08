'use strict'

import {MULTISAMPLE_COUNT, NUM_CELLS_X, NUM_CELLS_Z, CLOTH_SIDE_SIZE, NUM_ITERATIONS, GRAVITY} from './constants.js'
import { prepareDisplayShaderModule } from './displayModule.js';
import { prepareComputeShaderModule, projectStretchConstraint, projectBendConstraint } from './computeModule.js';

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

    const depthTextureFormat = 'depth32float';

    // ~~ SETUP GRAVITY SWITCH ~~
    let useGravity = false;
    const gravitySwitch = document.getElementById('gravitySwitch');

    gravitySwitch.addEventListener('change', function() {
        useGravity = this.checked; 
    });
    useGravity = gravitySwitch.checked;

    // ~~ SETUP VERTICES (position (vec4<f32>)) ~~
    const cellSideSizeX = CLOTH_SIDE_SIZE / NUM_CELLS_X;
    const cellSideSizeZ = CLOTH_SIDE_SIZE / NUM_CELLS_Z;
    //num of floats per vertex (XYZW) * num of vertices per row * num of vertices per column
    let vertices = new Float32Array(3 * (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1));
    let offset = 0;
    const addVertex = (x,z) => {
        vertices[offset++] = x;
        vertices[offset++] = 0.0;
        vertices[offset++] = z;
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
            format: "float32x3",
            }
        ],
        arrayStride: 12,
        stepMode: "vertex",
        },
    ];

    // ~~ SETUP INDICES ~~
    let col = NUM_CELLS_Z + 1;
    let getIdxByPos = (x,y) => {
        return x * col + y;
    };
    let indices = new Int32Array(6 * NUM_CELLS_X * NUM_CELLS_Z);
    let offsetIdx = 0;
    for (let w = 0; w < NUM_CELLS_X; ++w) {
        for (let h = 0; h < NUM_CELLS_Z; ++h) {
            indices[offsetIdx++] = getIdxByPos(w, h);
            indices[offsetIdx++] = getIdxByPos(w+1, h);
            indices[offsetIdx++] = getIdxByPos(w+1, h+1);
            indices[offsetIdx++] = getIdxByPos(w, h);
            indices[offsetIdx++] = getIdxByPos(w+1, h+1);
            indices[offsetIdx++] = getIdxByPos(w, h+1);
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
    
    // ~~ CREATE CONSTRAINTS  ~~

    //calculate reverse mass of the particles (mass is equal to the amount of triangles that particle belongs to)
    let verticesW = new Float32Array((NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1));
    for (let w = 0; w < NUM_CELLS_X + 1; ++w) {
        for (let h = 0; h < NUM_CELLS_Z + 1; ++h) {
            if ((w == 0 || w == NUM_CELLS_X) && (h == 0 || h == NUM_CELLS_Z))
                verticesW[getIdxByPos(w, h)] = 1.0 / 2.0;
            else if ((w == 0 || w == NUM_CELLS_X) || (h == 0 || h == NUM_CELLS_Z))
                verticesW[getIdxByPos(w, h)] = 1.0 / 3.0;
            else 
                verticesW[getIdxByPos(w, h)] = 1.0 / 6.0;
        }
    }
    
    //1 - simulate vertex using PBD, 0 - don't change it's position through PBD
    let simulate = new Int8Array((NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1));
    for (let i = 0; i < (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1); ++i) {
        simulate[i] = 1;
    }
    simulate[getIdxByPos(0,0)] = 0;
    simulate[getIdxByPos(NUM_CELLS_X,0)] = 0;
    simulate[getIdxByPos(0,NUM_CELLS_Z)] = 0;
    simulate[getIdxByPos(NUM_CELLS_X,NUM_CELLS_Z)] = 0;
    simulate[getIdxByPos(NUM_CELLS_X / 2, NUM_CELLS_Z / 2)] = 0;

    let velocities = new Float32Array(3 * (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1));
    for (let i = 0; i < 3 * (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1); ++i) {
        velocities[i] = 0.0;
    }

    let stretchConstr = [];
    for (let w = 0; w < NUM_CELLS_X + 1; ++w) {
        for (let h = 0; h < NUM_CELLS_Z + 1; ++h) {
            // *  `
            // |
            // *  `
            if (h < NUM_CELLS_Z) {
                let c = {};
                c.p1 = getIdxByPos(w, h);
                c.p2 = getIdxByPos(w, h + 1);
                c.l0 = cellSideSizeZ;
                stretchConstr.push(c);
            }
            // `  `
            // 
            // *--*
            if (w < NUM_CELLS_X) {
                let c = {};
                c.p1 = getIdxByPos(w, h);
                c.p2 = getIdxByPos(w + 1, h);
                c.l0 = cellSideSizeX;
                stretchConstr.push(c);
            }
            // `  *
            //  /
            // *  `
            if (w < NUM_CELLS_X && h < NUM_CELLS_Z) {
                let c = {};
                c.p1 = getIdxByPos(w, h);
                c.p2 = getIdxByPos(w + 1, h + 1);
                c.l0 = Math.sqrt(cellSideSizeX * cellSideSizeX + cellSideSizeZ * cellSideSizeZ);
                stretchConstr.push(c);
            }
        }
    }

    let bendConstr = [];
    for (let w = 0; w < NUM_CELLS_X; ++w) {
        for (let h = 0; h < NUM_CELLS_Z; ++h) {
            // *--*
            // |/ |
            // *--*
            {
                let c = {};
                c.p1 = getIdxByPos(w, h);
                c.p2 = getIdxByPos(w + 1, h + 1);
                c.p3 = getIdxByPos(w + 1, h);
                c.p4 = getIdxByPos(w, h + 1);
                c.phi = Math.PI;
                bendConstr.push(c);
            }
            // `  *
            //  / |
            // *--*
            // | /
            // *  `
            if (h > 0) {
                let c = {};
                c.p1 = getIdxByPos(w, h);
                c.p2 = getIdxByPos(w + 1, h);
                c.p3 = getIdxByPos(w + 1, h + 1);
                c.p4 = getIdxByPos(w, h - 1);
                c.phi = Math.PI;
                bendConstr.push(c);
            }
            // `  *--*
            //  / | /
            // *--*  `
            if (w > 0) {
                let c = {};
                c.p1 = getIdxByPos(w, h);
                c.p2 = getIdxByPos(w, h + 1);
                c.p3 = getIdxByPos(w - 1, h);
                c.p4 = getIdxByPos(w + 1, h + 1);
                c.phi = Math.PI;
                bendConstr.push(c);
            }
        }
    }

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
        depthStencil: {
            depthCompare: "less",
            depthWriteEnabled: true,
            format: depthTextureFormat,
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
        depthStencilAttachment: {
            depthClearValue: "1.0",
            depthLoadOp: "clear",
            depthStoreOp: "discard",
        }
    };

    let multisampleTexture;
    let depthTexture;
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

        if (!depthTexture || 
            depthTexture.width !== canvasTexture.width ||
            depthTexture.height !== canvasTexture.height) {

            // If we have an existing depth texture destroy it    
            if (depthTexture) {
                depthTexture.destroy();
            }

            depthTexture = device.createTexture({
                //always supported
                format: depthTextureFormat,
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
                size: [canvasTexture.width, canvasTexture.height],
                sampleCount: MULTISAMPLE_COUNT,
            });
        }

        device.queue.writeBuffer(computeSM.uniormBuffer, 0, new Float32Array([time, 0, 0, 0]), 0, 4);
        renderPassDescriptor.colorAttachments[0].view = multisampleTexture.createView();
        renderPassDescriptor.colorAttachments[0].resolveTarget = canvasTexture.createView();
        renderPassDescriptor.depthStencilAttachment.view = depthTexture.createView();
        {
            if (useGravity) {
                for (let i = 0; i < (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1); ++i) {
                    if (simulate[i]) {
                        velocities[3*i + 1] -= GRAVITY * deltaTime;
                    }
                }
            }

            vertices[getIdxByPos(NUM_CELLS_X / 2, NUM_CELLS_Z / 2) * 3 + 1] = Math.sin(time) * 0.5;
            for (let i = 0; i < (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1); ++i) {
                if (simulate[i]) {
                    for (let v = 0; v < 3; ++v) {
                        vertices[3*i + v] += velocities[3*i + v] * deltaTime;
                    }
                }
            }
            for (let i = 0; i < NUM_ITERATIONS; ++i) {
                bendConstr.forEach(c => {
                    let p1 = [vertices[c.p1 * 3], vertices[c.p1 * 3 + 1], vertices[c.p1 * 3 + 2]];
                    let p2 = [vertices[c.p2 * 3], vertices[c.p2 * 3 + 1], vertices[c.p2 * 3 + 2]];
                    let p3 = [vertices[c.p3 * 3], vertices[c.p3 * 3 + 1], vertices[c.p3 * 3 + 2]];
                    let p4 = [vertices[c.p4 * 3], vertices[c.p4 * 3 + 1], vertices[c.p4 * 3 + 2]];
                    let phi = c.phi;
                    let w1 = verticesW[c.p1];
                    let w2 = verticesW[c.p2];
                    let w3 = verticesW[c.p3];
                    let w4 = verticesW[c.p4];
                    let res = projectBendConstraint(p1, p2, p3, p4, phi, w1, w2, w3, w4);
                    for (let v = 0; v < 3; ++v) {
                        if (simulate[c.p1]) {
                            velocities[c.p1 * 3 + c] = (res[0][v] - vertices[c.p1 * 3 + v]) / deltaTime;
                            vertices[c.p1 * 3 + v] = res[0][v];
                        }
                        if (simulate[c.p2]) {
                            velocities[c.p2 * 3 + c] = (res[1][v] - vertices[c.p2 * 3 + v]) / deltaTime;
                            vertices[c.p2 * 3 + v] = res[1][v];
                        }
                        if (simulate[c.p3]) {
                            velocities[c.p3 * 3 + c] = (res[2][v] - vertices[c.p3 * 3 + v]) / deltaTime;
                            vertices[c.p3 * 3 + v] = res[2][v];
                        }
                        if (simulate[c.p4]) {
                            velocities[c.p4 * 3 + c] = (res[3][v] - vertices[c.p4 * 3 + v]) / deltaTime;
                            vertices[c.p4 * 3 + v] = res[3][v];
                        }
                    }
                });
                stretchConstr.forEach(c => {
                    let p1 = [vertices[c.p1 * 3], vertices[c.p1 * 3 + 1], vertices[c.p1 * 3 + 2]];
                    let p2 = [vertices[c.p2 * 3], vertices[c.p2 * 3 + 1], vertices[c.p2 * 3 + 2]];
                    let w1 = verticesW[c.p1];
                    let w2 = verticesW[c.p2];
                    let d = c.l0;
                    let res = projectStretchConstraint(p1, p2, d, w1, w2);
                    for (let v = 0; v < 3; ++v) {
                        if (simulate[c.p1]) {
                            velocities[c.p1 * 3 + c] = (res[0][v] - vertices[c.p1 * 3 + v]) / deltaTime;
                            vertices[c.p1 * 3 + v] = res[0][v];
                        }
                        if (simulate[c.p2]) {
                            velocities[c.p2 * 3 + c] = (res[1][v] - vertices[c.p2 * 3 + v]) / deltaTime;
                            vertices[c.p2 * 3 + v] = res[1][v];
                        }
                    }
                });
            }
        }
        device.queue.writeBuffer(vertexBuffer, 0, vertices, 0, vertices.length);
        const commandEncoder = device.createCommandEncoder();

        // const computePassEncoder = commandEncoder.beginComputePass();
        // computePassEncoder.setPipeline(computePipeline);
        // computeSM.bindGroups.forEach((bg, idx) => {
        //     computePassEncoder.setBindGroup(idx, bg);
        // });
        // computePassEncoder.dispatchWorkgroups((NUM_CELLS_Z + 1) * (NUM_CELLS_X + 1));
        // computePassEncoder.end();

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