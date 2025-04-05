# General
1. Single core.
2. Load store arch.
3. Explicit local memory.
4. Static inst. sched.
5. Software managed latency
6. Platform has shared mem with coprocs operating through it
7. 32-entry GP regfile (rotatable?)
8. Var. length instrs (16, 32, 64)
9. Instruction packing is restricted (alignment, slots)


# Mem hierarchy
1. Shmem is a top level memory. Each coprocessor can access shmem.
2. Private mem. Local to coprocessors.

# Instr Format
1. Instructions are grouped in 64 bit packs. (4 16bit instrs in each pack)
2. Jumps are only allowed to 8-aligned addrs.
3. var length insns must be aligned within the pack.
4. insns must be put in appropriate slots within the pack or they will be ignored
5. only first slot can contain metadata
6. only last slot can contain control transfer instructions

```
I1: (opcode)(REG_ARG_1)
I2: (opcode)(REG_ARG_1)(REG_ARG_1)
( 4)( 3)( 2)( 1)
(16)(16)(16)(16)
(AL)(AL)(AL)(AL) AL - arith and logic
(LM)(LM)(LM)(LM) LM - local motion (reg file copies), immidiates
    (LS)(LS)     LS - loads & stores
            (MD) MD - metadata
(CT)             CT - control transfer
```


### Types
1. load/store/copy
   1. LDS - issues a segment load op to the mem subsystem (completion has to be checked by software)
   2. STS - store segment (vends an op token)
   3. STSU - store segment (no token vend)
   4. LDV - load a value from local mem
   5. STV - store value to local mem
2. alu ops
   1. SUMM - addition
   2. AI - addition complement
   3. AND
   4. OR
   5. NOT
3. control
   1. GOTO - ip += imm offset (happens at the frontend, which means that local transfers are instant)
   2. DTC - dispatch transfer control
      1. DTCIOC - dispatch immidiate offset conditional
      2. DTCRO - dispatch register offset
      3. DTCROC - dispatch register offset conditional
      4. DTCRAA - dispatch register absolute address
      5. DTCRAAC - dispatch register absolute address conditional
   3. TC - transfer control (stalls frontend if dispatched control transfer has not retired)
4. special
   1. CHKICOPS - check issued coprocessor operation status
   2. PUC - put constant into register
   3. CPY - reg-reg copy
   4. DNO - do nothing
   5. SRS - switch register set