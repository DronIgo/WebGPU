import { mat4, vec3 } from 'https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.js';

const readFile = (filename) => {
    const reader = new FileReader();
    let text;
    reader.onload = () => {
        text = reader.result;
    };
    reader.onerror = () => {
        console.error("failed to read file ", filename)
    };
    reader.readAsText(filename)
}

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

    const multisampleCount = 4;

    // ~~ SETUP VERTICES (position (vec3<f32>), color(vec3<f32>)) ~~
    // Pack them all into one array
    // Each vertex has a position and a color packed in memory in X Y Z R G B order
    const cell_width = 4;
    const cell_height = 4;
    const cell_side_size = 1.0;
    const color = [1.0, 0.0, 0.5];
    //num of floats per vertex (XYZRGB) * num of vertex per cell * num of cells
    let vertices = new Float32Array(8 * 6 * cell_width * cell_height);
    let offset = 0;
    const addVertex = (x,z) => {
        vertices[offset++] = x;
        vertices[offset++] = 0.0;
        vertices[offset++] = z;
        vertices[offset++] = 1.0;
        vertices[offset++] = color[0];
        vertices[offset++] = color[1];
        vertices[offset++] = color[2];
        vertices[offset++] = 1.0;
    };
    for (let w = 0; w < cell_width; w++) {
        for (let h = 0; h < cell_height; h++) {
            const l_x = (w - cell_width/2) * cell_side_size;
            const r_x = (w + 1 - cell_width/2) * cell_side_size;
            const n_z = (h - cell_height/2) * cell_side_size;
            const f_z = (h + 1 - cell_height/2) * cell_side_size;
            //bottom triangle
            addVertex(l_x, n_z);
            addVertex(r_x, n_z);
            addVertex(r_x, f_z);
            //top triangle
            addVertex(l_x, n_z);
            addVertex(r_x, f_z);
            addVertex(l_x, f_z);
            console.debug("triangle");
            console.debug(l_x, " ", n_z);
            console.debug(r_x, " ", f_z);
            console.debug(r_x, " ", n_z);

            console.debug("triangle");
            console.debug(l_x, " ", n_z);
            console.debug(l_x, " ", f_z);
            console.debug(r_x, " ", f_z);
        }
    }
    const vertexBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
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
            },
            {
            shaderLocation: 1,
            offset: 16,
            format: "float32x4",
            },
        ],
        arrayStride: 32,
        stepMode: "vertex",
        },
    ];

    // ~~ SETUP UNIFORM BUFFER ~~
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

    const uniformBuffer = device.createBuffer({
        size: mvpMatrix.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(uniformBuffer.getMappedRange()).set(mvpMatrix);
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

    // ~~ DEFINE BASIC SHADERS ~~
    //TO DO: find a way to load from a separate file
    const displayCode = `
    struct VertexOut {
        @builtin(position) position : vec4<f32>,
        @location(0) color : vec4<f32>,
    };

    @group(0) @binding(0) var<uniform> mvp : mat4x4<f32>;

    @vertex
    fn vertex_main(
        @location(0) position: vec4<f32>,
        @location(1) color: vec3<f32>
    ) -> VertexOut
    {
        var output : VertexOut;
        output.position = mvp * position;
        output.color = vec4(color, 1.0);
        return output;
    } 

    @fragment
    fn fragment_main(fragData: VertexOut) -> @location(0) vec4<f32>
    {
        return fragData.color;
    } `;
    
    const shaderModule = device.createShaderModule({
        code: displayCode
    });

    // ~~ CREATE RENDER PIPELINE ~~
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });

    const pipeline = device.createRenderPipeline({
        vertex: {
        module: shaderModule,
        entryPoint: "vertex_main",
        buffers: vertexBuffersDescriptors,
        },
        fragment: {
        module: shaderModule,
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

        renderPassDescriptor.colorAttachments[0].view = multisampleTexture.createView();
        renderPassDescriptor.colorAttachments[0].resolveTarget = canvasTexture.createView();
        const commandEncoder = device.createCommandEncoder();
        const passEncoder =
        commandEncoder.beginRenderPass(renderPassDescriptor);

        passEncoder.setPipeline(pipeline);
        passEncoder.setVertexBuffer(0, vertexBuffer);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.draw(cell_width * cell_height * 2 * 3);
        passEncoder.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
};

init();