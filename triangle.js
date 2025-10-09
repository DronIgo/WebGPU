'use strict'

import {MULTISAMPLE_COUNT, NUM_CELLS_X, NUM_CELLS_Z, CLOTH_SIDE_SIZE, NUM_ITERATIONS, GRAVITY} from './constants.js'
import { prepareDisplayShaderModule } from './displayModule.js';
import { prepareComputeShaderModule, projectStretchConstraint, projectBendConstraint } from './computeModule.js';
import { generateConstraintBindGroups, createIntBuffer, createFloatBuffer } from './constraints.js';

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
    //we will use two separate arrays (and two separate buffers for vertices) to improve speed
    //this way compute shader can run in parallel to display shader
    let vertices = new Float32Array(3 * (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1));
    let vertices2 = new Float32Array(3 * (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1));
    let offset = 0;
    let offset2 = 0;
    const addVertex = (x,z) => {
        vertices[offset++] = x;
        vertices[offset++] = 0.0;
        vertices[offset++] = z;
        vertices2[offset2++] = x;
        vertices2[offset2++] = 0.0;
        vertices2[offset2++] = z;
    };
    for (let w = 0; w < NUM_CELLS_X + 1; w++) {
        for (let h = 0; h < NUM_CELLS_Z + 1; h++) {
            const x = (w - NUM_CELLS_X/2) * cellSideSizeX;
            const z = (h - NUM_CELLS_Z/2) * cellSideSizeZ;
            addVertex(x, z);
        }
    }
    const vertexBuffer = createFloatBuffer(device, vertices, "vertex buffer", {s:true, v:true});

    const vertexBuffer2 = createFloatBuffer(device, vertices2, "vertex buffer 2", {s:true, v:true});

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

    let indexBuffer = createIntBuffer(device, indices, "index buffer", {i : true});
    
    // ~~ CREATE NEEDED DATA BUFFERS ~~
    //calculate reverse mass of the particles (mass is equal to the amount of triangles that particle belongs to)
    let invMass = new Float32Array((NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1));
    for (let w = 0; w < NUM_CELLS_X + 1; ++w) {
        for (let h = 0; h < NUM_CELLS_Z + 1; ++h) {
            if ((w == 0 || w == NUM_CELLS_X) && (h == 0 || h == NUM_CELLS_Z))
                invMass[getIdxByPos(w, h)] = 1.0 / 2.0;
            else if ((w == 0 || w == NUM_CELLS_X) || (h == 0 || h == NUM_CELLS_Z))
                invMass[getIdxByPos(w, h)] = 1.0 / 3.0;
            else 
                invMass[getIdxByPos(w, h)] = 1.0 / 6.0;
        }
    }
    
    //1 - simulate vertex using PBD, 0 - don't change it's position through PBD
    let simulate = new Int32Array((NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1));
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

    let velocitiesBuffer = createFloatBuffer(device, velocities, "velocities buffer", {s:true});

    let simulatedBuffer = createIntBuffer(device, simulate, "simulated buffer", {s : true});

    let invMassBuffer = createFloatBuffer(device, invMass, "inv mass buffer", {s : true});

    // ~~ PREPARE DISPLAY SHADER MODULE (as well as all related uniform buffers, bind group layouts and bind groups) ~~
    let dispSM = prepareDisplayShaderModule(device, canvas.clientWidth / canvas.clientHeight, NUM_CELLS_Z);

    // ~~ PREPARE COMPUTE PIPELINES (as well as all related uniform buffers, bind group layouts and bind groups) ~~
    let compute = prepareComputeShaderModule(device, CLOTH_SIDE_SIZE, NUM_CELLS_X, NUM_CELLS_Z);

    let bindGroupSimulatedInvMass = device.createBindGroup({
        layout: compute.bindGroupLayoutRSRS,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: simulatedBuffer
                }
            },
            { 
            binding: 1, 
            resource:  
                { 
                buffer: invMassBuffer
                }
            },
        ],
    });

    let bindGroupVelocitiesReadOnly = device.createBindGroup({
        layout: compute.bindGroupLayoutRS,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: velocitiesBuffer
                }
            }
        ],
    });

    let bindGroupVelocities = device.createBindGroup({
        layout: compute.bindGroupLayoutS,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: velocitiesBuffer
                }
            }
        ],
    });

    let bindGroupVertR1W2 = device.createBindGroup({
        layout: compute.bindGroupLayoutRSS,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: vertexBuffer
                }
            },
            { 
            binding: 1, 
            resource:  
                { 
                buffer: vertexBuffer2
                }
            },
        ],
    });

    let bindGroupVertR2W1 = device.createBindGroup({
        layout: compute.bindGroupLayoutRSS,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: vertexBuffer2
                }
            },
            { 
            binding: 1, 
            resource:  
                { 
                buffer: vertexBuffer
                }
            },
        ],
    });

        let bindGroupVertP1C2 = device.createBindGroup({
        layout: compute.bindGroupLayoutRSRS,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: vertexBuffer
                }
            },
            { 
            binding: 1, 
            resource:  
                { 
                buffer: vertexBuffer2
                }
            },
        ],
    });

    let bindGroupVertP2C1 = device.createBindGroup({
        layout: compute.bindGroupLayoutRSRS,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: vertexBuffer2
                }
            },
            { 
            binding: 1, 
            resource:  
                { 
                buffer: vertexBuffer
                }
            },
        ],
    });

    let bindGroupVert1 = device.createBindGroup({
        layout: compute.bindGroupLayoutS,
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

    let bindGroupVert2 = device.createBindGroup({
        layout: compute.bindGroupLayoutS,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: vertexBuffer2
                }
            },
        ],
    });

    // ~~ CREATE CONSTRAINTS ~~
    let constrBG = generateConstraintBindGroups(device, NUM_CELLS_X, NUM_CELLS_Z, CLOTH_SIDE_SIZE, getIdxByPos, compute.bindGroupLayoutRSRSU);

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
    let writeToVertices2 = false;
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
        //using fixed delta time 
        deltaTime = 1.0 / 20.0;

        writeToVertices2 = !writeToVertices2;
        let verticesRead = vertices2;
        let verticesWrite = vertices;
        let bgVerticesRW = bindGroupVertR2W1;
        let bgVerticesCP = bindGroupVertP2C1;
        let bgVerticesComp = bindGroupVert1;
        let vertexBufferDisp = vertexBuffer2;
        if (writeToVertices2) {
            verticesRead = vertices;
            verticesWrite = vertices2;
            bgVerticesRW = bindGroupVertR1W2;
            bgVerticesCP = bindGroupVertP1C2;
            bgVerticesComp = bindGroupVert2;
            vertexBufferDisp = vertexBuffer
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

        //device.queue.writeBuffer(computeSM.uniormBuffer, 0, new Float32Array([time, 0, 0, 0]), 0, 4);
        renderPassDescriptor.colorAttachments[0].view = multisampleTexture.createView();
        renderPassDescriptor.colorAttachments[0].resolveTarget = canvasTexture.createView();
        renderPassDescriptor.depthStencilAttachment.view = depthTexture.createView();
        // {
        //     verticesWrite[getIdxByPos(NUM_CELLS_X / 2, NUM_CELLS_Z / 2) * 3 + 1] = Math.sin(time) * 0.5;
            
        //     if (useGravity) {
        //         for (let i = 0; i < (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1); ++i) {
        //             if (simulate[i]) {
        //                 velocities[3*i + 1] -= GRAVITY * deltaTime * invMass[i];
        //             }
        //         }
        //     }

        //     for (let i = 0; i < (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1); ++i) {
        //         if (simulate[i]) {
        //             for (let v = 0; v < 3; ++v) {
        //                 verticesWrite[3*i + v] = verticesRead[3*i + v] + velocities[3*i + v] * deltaTime;
        //             }
        //         }
        //     }

        //     for (let i = 0; i < NUM_ITERATIONS; ++i) {
        //         bendConstr.forEach(c => {
        //            let p1 = [verticesWrite[c.p1 * 3], verticesWrite[c.p1 * 3 + 1], verticesWrite[c.p1 * 3 + 2]];
        //            let p2 = [verticesWrite[c.p2 * 3], verticesWrite[c.p2 * 3 + 1], verticesWrite[c.p2 * 3 + 2]];
        //            let p3 = [verticesWrite[c.p3 * 3], verticesWrite[c.p3 * 3 + 1], verticesWrite[c.p3 * 3 + 2]];
        //            let p4 = [verticesWrite[c.p4 * 3], verticesWrite[c.p4 * 3 + 1], verticesWrite[c.p4 * 3 + 2]];
        //            let phi = c.phi;
        //            let w1 = invMass[c.p1];
        //            let w2 = invMass[c.p2];
        //            let w3 = invMass[c.p3];
        //            let w4 = invMass[c.p4];
        //            let s1 = simulate[c.p1];
        //            let s2 = simulate[c.p2];
        //            let s3 = simulate[c.p3];
        //            let s4 = simulate[c.p4];
        //            let res = projectBendConstraint(p1, p2, p3, p4, phi, w1, w2, w3, w4, s1, s2, s3, s4);
        //            for (let v = 0; v < 3; ++v) {
        //                if (simulate[c.p1])
        //                    verticesWrite[c.p1 * 3 + v] = res[0][v];
        //                if (simulate[c.p2])
        //                    verticesWrite[c.p2 * 3 + v] = res[1][v];
        //                if (simulate[c.p3])
        //                    verticesWrite[c.p3 * 3 + v] = res[2][v];
        //                if (simulate[c.p4]) 
        //                    verticesWrite[c.p4 * 3 + v] = res[3][v];
        //            }
        //         });
        //         stretchConstr.forEach(c => {
        //             let p1 = [verticesWrite[c.p1 * 3], verticesWrite[c.p1 * 3 + 1], verticesWrite[c.p1 * 3 + 2]];
        //             let p2 = [verticesWrite[c.p2 * 3], verticesWrite[c.p2 * 3 + 1], verticesWrite[c.p2 * 3 + 2]];
        //             let w1 = invMass[c.p1];
        //             let w2 = invMass[c.p2];
        //             let d = c.l0;
        //             let s1 = simulate[c.p1];
        //             let s2 = simulate[c.p2];
        //             let res = projectStretchConstraint(p1, p2, d, w1, w2, s1, s2);
        //             for (let v = 0; v < 3; ++v) {
        //                 if (simulate[c.p1])
        //                     verticesWrite[c.p1 * 3 + v] = res[0][v];
        //                 if (simulate[c.p2])
        //                     verticesWrite[c.p2 * 3 + v] = res[1][v];
        //             }
        //         });
        //     }
        //     for (let i = 0; i < (NUM_CELLS_X + 1) * (NUM_CELLS_Z + 1); ++i) {
        //         if (simulate[i]) {
        //             for (let v = 0; v < 3; ++v)
        //                 velocities[3*i+v] = (verticesRead[3*i+v] - verticesWrite[3*i+v]) / deltaTime;
        //         }
        //     }
        //     device.queue.writeBuffer(vertexBuffer, 0, verticesRead, 0, vertices.length);
        // }

        let gravity = useGravity ? 1.0 : 0.0;
        device.queue.writeBuffer(compute.perFrameUniormBuffer, 0, new Float32Array([deltaTime, gravity, 0, 0]), 0, 4);

        const commandEncoder = device.createCommandEncoder();

        const computePassEncoder = commandEncoder.beginComputePass();
        //Update positions 
        computePassEncoder.setPipeline(compute.updateInitPipeline);
        computePassEncoder.setBindGroup(0, bgVerticesRW);
        computePassEncoder.setBindGroup(1, bindGroupSimulatedInvMass);
        computePassEncoder.setBindGroup(2, bindGroupVelocitiesReadOnly);
        computePassEncoder.setBindGroup(3, compute.bindGroupPerFrame);
        computePassEncoder.dispatchWorkgroups(Math.ceil((NUM_CELLS_Z + 1) * (NUM_CELLS_X + 1) / 64.0));

        //Solve Gauss-Seidel for constraaints
        for (let i = 0; i < NUM_ITERATIONS; ++i) {
            computePassEncoder.setPipeline(compute.bendPipeline);
            computePassEncoder.setBindGroup(0, bgVerticesComp);
            computePassEncoder.setBindGroup(1, bindGroupSimulatedInvMass);
            constrBG.bend.forEach((bg) => {
                computePassEncoder.setBindGroup(2, bg.bg);
                computePassEncoder.dispatchWorkgroups(Math.ceil(bg.numInv / 64.0));
            });

            computePassEncoder.setPipeline(compute.stretchPipeline);
            computePassEncoder.setBindGroup(0, bgVerticesComp);
            computePassEncoder.setBindGroup(1, bindGroupSimulatedInvMass);
            constrBG.stretch.forEach((bg) => {
                computePassEncoder.setBindGroup(2, bg.bg);
                computePassEncoder.dispatchWorkgroups(Math.ceil(bg.numInv / 64.0));
            });
        }

        computePassEncoder.setPipeline(compute.updateFinalPipeline);
        computePassEncoder.setBindGroup(0, bgVerticesCP);
        computePassEncoder.setBindGroup(2, bindGroupVelocities);
        computePassEncoder.setBindGroup(3, compute.bindGroupPerFrame);
        computePassEncoder.dispatchWorkgroups(Math.ceil((NUM_CELLS_Z + 1) * (NUM_CELLS_X + 1) / 64.0));

        computePassEncoder.end();

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

        passEncoder.setPipeline(pipeline);
        passEncoder.setVertexBuffer(0, vertexBufferDisp);
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