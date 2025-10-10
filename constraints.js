export function createIntBuffer(dev, arr, label, usage) {
    let gpuUsage = GPUBufferUsage.COPY_DST;
    if (usage.s) {
        gpuUsage |= GPUBufferUsage.STORAGE;
    }
    if (usage.u) {
        gpuUsage |= GPUBufferUsage.UNIFORM;
    }
    if (usage.v) {
        gpuUsage |= GPUBufferUsage.VERTEX;
    }
    if (usage.i) {
        gpuUsage |= GPUBufferUsage.INDEX;
    }
    const buffer = dev.createBuffer({
        label: label,
        size: arr.byteLength,
        usage: gpuUsage,
        mappedAtCreation: true,
    });
    new Int32Array(buffer.getMappedRange()).set(arr);
    buffer.unmap();
    return buffer;
};

export function createUintBuffer(dev, arr, label, usage) {
    let gpuUsage = GPUBufferUsage.COPY_DST;
    if (usage.s) {
        gpuUsage |= GPUBufferUsage.STORAGE;
    }
    if (usage.u) {
        gpuUsage |= GPUBufferUsage.UNIFORM;
    }
    if (usage.v) {
        gpuUsage |= GPUBufferUsage.VERTEX;
    }
    if (usage.i) {
        gpuUsage |= GPUBufferUsage.INDEX;
    }
    const buffer = dev.createBuffer({
        label: label,
        size: arr.byteLength,
        usage: gpuUsage,
        mappedAtCreation: true,
    });
    new Uint32Array(buffer.getMappedRange()).set(arr);
    buffer.unmap();
    return buffer;
};

export function createFloatBuffer(dev, arr, label, usage) {
    let gpuUsage = GPUBufferUsage.COPY_DST;
    if (usage.s) {
        gpuUsage |= GPUBufferUsage.STORAGE;
    }
    if (usage.u) {
        gpuUsage |= GPUBufferUsage.UNIFORM;
    }
    if (usage.v) {
        gpuUsage |= GPUBufferUsage.VERTEX;
    }
    if (usage.i) {
        gpuUsage |= GPUBufferUsage.INDEX;
    }
    const buffer = dev.createBuffer({
        label: label,
        size: arr.byteLength,
        usage: gpuUsage,
        mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(arr);
    buffer.unmap();
    return buffer;
};

function stretchArrayToBindGroup(device, arr, bindGroupLayoutRSRSU) {
    let arrIdx = new Uint32Array(arr.length * 2);
    let arrL0 = new Float32Array(arr.length);
    let numConstr = new Uint32Array(4);
    numConstr[0] = arr.length;
    numConstr[1] = 0;
    numConstr[2] = 0;
    numConstr[3] = 0;
    for (let i = 0; i < arr.length; ++i) {
        arrIdx[2*i] = arr[i].p1;
        arrIdx[2*i+1] = arr[i].p2;
        arrL0[i] = arr[i].l0;
    }
    let idxBuffer = createUintBuffer(device, arrIdx, "", {s:true});
    let l0Buffer = createFloatBuffer(device, arrL0, "", {s:true});
    let numBuffer = createUintBuffer(device, numConstr, "", {u:true});
    let bindGroup = device.createBindGroup({
        layout: bindGroupLayoutRSRSU,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: idxBuffer
                }
            },
            { 
            binding: 1, 
            resource:  
                { 
                buffer: l0Buffer
                }
            },
            { 
            binding: 2, 
            resource:  
                { 
                buffer: numBuffer
                }
            },
        ],
    });
    return {bg : bindGroup, numInv: arr.length};
}

function bendArrayToBindGroup(device, arr, bindGroupLayoutRSRSU) {
    let arrIdx = new Uint32Array(arr.length * 4);
    let arrPhi = new Float32Array(arr.length);
    let numConstr = new Uint32Array(4);
    numConstr[0] = arr.length;
    numConstr[1] = 0;
    numConstr[2] = 0;
    numConstr[3] = 0;
    for (let i = 0; i < arr.length; ++i) {
        arrIdx[4*i] = arr[i].p1;
        arrIdx[4*i+1] = arr[i].p2;
        arrIdx[4*i+2] = arr[i].p3;
        arrIdx[4*i+3] = arr[i].p4;
        arrPhi[i] = arr[i].phi;
    }
    let idxBuffer = createUintBuffer(device, arrIdx, "", {s:true});
    let phiBuffer = createFloatBuffer(device, arrPhi, "", {s:true});
    let numBuffer = createUintBuffer(device, numConstr, "", {u:true});
    let bindGroup = device.createBindGroup({
        layout: bindGroupLayoutRSRSU,
        entries: [
            { 
            binding: 0, 
            resource:  
                { 
                buffer: idxBuffer
                }
            },
            { 
            binding: 1, 
            resource:  
                { 
                buffer: phiBuffer
                }
            },
            { 
            binding: 2, 
            resource:  
                { 
                buffer: numBuffer
                }
            },
        ],
    });
    return {bg : bindGroup, numInv: arr.length};
}

//All PBD constraints are separated into 18 groups so that constraints within each group don't have any common points
//Each group can be run in parallel and in total this scheme requires 18 dispatches
//This IS suboptimal, better option would be to separate all constraint into 4 groups of 2x2 blocks like this:
// 0011
// 0011
// 2233
// 2233
//This scheme only takes 4 dispatches and is way better, but would require some refactoring and I simply don't have enough time
export function generateConstraintBindGroups(device, NUM_CELLS_X, NUM_CELLS_Z, CLOTH_SIDE_SIZE, getIdxByPos, bindGroupLayoutRSRSU) {
    const cellSideSizeX = CLOTH_SIDE_SIZE / NUM_CELLS_X;
    const cellSideSizeZ = CLOTH_SIDE_SIZE / NUM_CELLS_Z;

    let stretchConstrVert0 = [];
    let stretchConstrVert1 = [];
    let stretchConstrHor0 = [];
    let stretchConstrHor1 = [];
    let stretchConstrDiag0 = [];
    let stretchConstrDiag1 = [];
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
                if (h % 2 == 0)
                    stretchConstrHor0.push(c);
                else
                    stretchConstrHor1.push(c);
            }
            // `  `
            // 
            // *--*
            if (w < NUM_CELLS_X) {
                let c = {};
                c.p1 = getIdxByPos(w, h);
                c.p2 = getIdxByPos(w + 1, h);
                c.l0 = cellSideSizeX;
                if (w % 2 == 0)
                    stretchConstrVert0.push(c);
                else
                    stretchConstrVert1.push(c);
            }
            // `  *
            //  /
            // *  `
            if (w < NUM_CELLS_X && h < NUM_CELLS_Z) {
                let c = {};
                c.p1 = getIdxByPos(w, h);
                c.p2 = getIdxByPos(w + 1, h + 1);
                c.l0 = Math.sqrt(cellSideSizeX * cellSideSizeX + cellSideSizeZ * cellSideSizeZ);
                if (w % 2 == 0)
                    stretchConstrDiag0.push(c);
                else
                    stretchConstrDiag1.push(c);
            }
        }
    }
    
    let bendConstrCenter00 = [];
    let bendConstrCenter01 = [];
    let bendConstrCenter10 = [];
    let bendConstrCenter11 = [];

    let bendConstrLeft00 = [];
    let bendConstrLeft01 = [];
    let bendConstrLeft10 = [];
    let bendConstrLeft11 = [];

    let bendConstrDown00 = [];
    let bendConstrDown01 = [];
    let bendConstrDown10 = [];
    let bendConstrDown11 = [];
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
                if (w % 2 == 0) 
                    if (h % 2 == 0)
                        bendConstrCenter00.push(c);
                    else
                        bendConstrCenter01.push(c);
                else
                    if (h % 2 == 0)
                        bendConstrCenter10.push(c);
                    else
                        bendConstrCenter11.push(c);
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
                if (w % 2== 0) 
                    if (h % 2== 0)
                        bendConstrDown00.push(c);
                    else
                        bendConstrDown01.push(c);
                else
                    if (h % 2 == 0)
                        bendConstrDown10.push(c);
                    else
                        bendConstrDown11.push(c);
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
                if (w % 2== 0) 
                    if (h % 2 == 0)
                        bendConstrLeft00.push(c);
                    else
                        bendConstrLeft01.push(c);
                else
                    if (h % 2 == 0)
                        bendConstrLeft10.push(c);
                    else
                        bendConstrLeft11.push(c);
            }
        }
    }
    
    let bgSV0 = stretchArrayToBindGroup(device, stretchConstrVert0, bindGroupLayoutRSRSU);
    let bgSV1 = stretchArrayToBindGroup(device, stretchConstrVert1, bindGroupLayoutRSRSU);
    let bgSH0 = stretchArrayToBindGroup(device, stretchConstrHor0, bindGroupLayoutRSRSU);
    let bgSH1 = stretchArrayToBindGroup(device, stretchConstrHor1, bindGroupLayoutRSRSU);
    let bgSD0 = stretchArrayToBindGroup(device, stretchConstrDiag0, bindGroupLayoutRSRSU);
    let bgSD1 = stretchArrayToBindGroup(device, stretchConstrDiag1, bindGroupLayoutRSRSU);

    let bgBC00 = bendArrayToBindGroup(device, bendConstrCenter00, bindGroupLayoutRSRSU);
    let bgBC01 = bendArrayToBindGroup(device, bendConstrCenter01, bindGroupLayoutRSRSU);
    let bgBC10 = bendArrayToBindGroup(device, bendConstrCenter10, bindGroupLayoutRSRSU);
    let bgBC11 = bendArrayToBindGroup(device, bendConstrCenter11, bindGroupLayoutRSRSU);

    let bgBD00 = bendArrayToBindGroup(device, bendConstrDown00, bindGroupLayoutRSRSU);
    let bgBD01 = bendArrayToBindGroup(device, bendConstrDown01, bindGroupLayoutRSRSU);
    let bgBD10 = bendArrayToBindGroup(device, bendConstrDown10, bindGroupLayoutRSRSU);
    let bgBD11 = bendArrayToBindGroup(device, bendConstrDown11, bindGroupLayoutRSRSU);

    let bgBL00 = bendArrayToBindGroup(device, bendConstrLeft00, bindGroupLayoutRSRSU);
    let bgBL01 = bendArrayToBindGroup(device, bendConstrLeft01, bindGroupLayoutRSRSU);
    let bgBL10 = bendArrayToBindGroup(device, bendConstrLeft10, bindGroupLayoutRSRSU);
    let bgBL11 = bendArrayToBindGroup(device, bendConstrLeft11, bindGroupLayoutRSRSU);

    let res = {
        stretch: [bgSV0, bgSV1, bgSH0, bgSH1, bgSD0, bgSD1],
        bend : [bgBD00, bgBD01, bgBD10, bgBD11,
                bgBL00, bgBL01, bgBL10, bgBL11,
                bgBC00, bgBC01, bgBC10, bgBC11],
    };
    return res;
}