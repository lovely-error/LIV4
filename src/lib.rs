#![feature(generic_arg_infer)]
#![feature(str_from_raw_parts)]
#![feature(portable_simd)]
#![feature(decl_macro)]
#![feature(iter_from_coroutine)]
#![feature(coroutines)]

mod logical_model;

use core::{alloc::Layout, hint::unreachable_unchecked, mem::{transmute, ManuallyDrop, MaybeUninit}, ops::Range, ptr::{copy_nonoverlapping, null_mut}, sync::atomic::{fence, AtomicBool, AtomicU32, Ordering}};
use std::{alloc::alloc_zeroed, simd::{cmp::SimdPartialEq, num::SimdUint}};
use core::simd::Simd;
use atomic_spsc_queue::RingQueue;


const OPCODE_LIMIT: u8 = 32;

#[allow(non_camel_case_types)]
#[repr(u8)]
enum Opcode {

    DNO = 0,

    LDS = 1, // load segment
    STS = 2, // store segment
    LDV = 3, // load value
    STV = 4, // store value

    ADD = 5, //
    COMPL = 6, // additive inverse
    AND = 7,
    OR = 8,
    NOT = 9,

    PCT_GOTO16 = 10, // dispatched goto unconditional register offset or absolute address
    GOTO = 11, // goto unconditional immidiate offset

    CHKICOS = 12, // check issued coprocessor operation status
    PUC16 = 13, // put small immidiate
    PUC32 = 14, // put larger immidiate
    CPY = 15, // reg reg copy

    SDM = 16, // special decode marker. currently used only for split patterns, may be used for fused alu operations

    SRB = 17, // set register base

    PCT_GOTOIOC = 18, // dispatched goto conditional immidiate offset

    TC = 19, // transfer control

    ICO = 20, // issue coprocessor operation

    SEL = 21, // conditional select (cond ? a : b)
    STVC = 22, // store conditional

    MUL = 23,
    SHL = 24,
    SHR = 25,

    PREFIX_I64 = 26,

    HALT = 31,

    OPCODE_LIMIT = OPCODE_LIMIT
}

#[allow(non_camel_case_types)]
enum PackFormat {
    I16x4,
    I16x2_I32x1,
    I32x2,
    // I64x1,
}

// impl <T> AddAssign<usize> for *mut T {
//     fn add_assign(&mut self, rhs: usize) {
//         unsafe { *self = self.add(rhs) }
//     }
// }

fn compute_pack_format(pack: u64) -> PackFormat {
    let ptr = &raw const pack;
    let mut ptr = ptr.cast::<u16>();
    let opcode = read_opcode(ptr.cast());
    let match_first_32 =
        matches!(opcode, Opcode::GOTO) ||
        matches!(opcode, Opcode::PUC32) ;
    ptr = unsafe { ptr.add(2) };
    let opcode = read_opcode(ptr.cast());
    let match_second_32 =
        matches!(opcode, Opcode::GOTO) ||
        matches!(opcode, Opcode::PUC32) ||
        matches!(opcode, Opcode::PCT_GOTOIOC);
    match (match_first_32, match_second_32) {
        (true, true) => {
            return PackFormat::I32x2;
        },
        (true, false) => panic!("invalid format"),
        (false, true) => {
            return PackFormat::I16x2_I32x1;
        },
        (false, false) => {
            return PackFormat::I16x4;
        },
    }
}



fn read_opcode(insn: *const u8) -> Opcode {
    let val = unsafe { insn.read() };
    let ins = val & ((1 << 5) - 1);
    unsafe { transmute(ins) }
}
fn read_i16_alu_op2_args(insn: u16) -> (u8, u8) {
    let mut insn = insn >> 5;
    let rd = insn & ((1 << 5) - 1);
    insn = insn >> 5;
    let rs = insn & ((1 << 5) - 1);
    return (rd as _, rs as _)
}
fn read_i16_alu_op1_arg(insn: u16) -> u8 {
    let insn = insn >> 5;
    let rd = insn & ((1 << 5) - 1);
    return rd as _
}
fn read_bit_range_u32(
    num:u32,
    range: Range<usize>
) -> u32 {
    let low = range.start;
    let span = (1 << (range.end - low)) - 1;
    (num >> low) & span
}

#[test] #[ignore]
fn bitrange() {
    let num = 4293951570;
    let range = 10..15;
    let val = read_bit_range_u32(num, range.clone());
    println!("{}", val);
}

const OPCODE_HEADER1_BITWIDTH: usize = 5;


struct PUCS(u16);
impl PUCS {
    fn new(
        rd: u8,
        imm: u8,
    ) -> Self {
        assert!(rd < 32);
        let imm = imm as u16;
        if imm > ((1 << 6) - 1) {
            panic!("invalid value")
        }
        let val = Opcode::PUC16 as u16 |
            ((rd as u16) << OPCODE_HEADER1_BITWIDTH) |
            ((imm) << (OPCODE_HEADER1_BITWIDTH + 5));
        Self(val)
    }
    fn to_raw(self) -> u16 { self.0 }
}
struct PUCE(u32);
impl PUCE {
    fn to_raw(self) -> u32 {
        self.0
    }
    fn new(
        rd: u8,
        imm: u32
    ) -> Self {
        assert!(rd < 32);
        assert!(imm < (1 << 22));
        let val = (Opcode::PUC32 as u32) | ((rd as u32) << OPCODE_HEADER1_BITWIDTH) | (imm << 10);
        Self(val)
    }
}

#[derive(Debug, Clone, Copy)]
struct CPY(u16);
impl CPY {
    fn new(rd:u8, rs: u8) -> Self {
        assert!(rd < 32 && rs < 32);
        if rs >= 32 || rd >= 32 { panic!("invalid input") }
        let val = Opcode::CPY as u16 |
                  ((rd as u16) << OPCODE_HEADER1_BITWIDTH) |
                  ((rs as u16) << (5 + 5));
        return Self(val)
    }
    /// (rd, rs)
    fn unpack(&self) -> (u8, u8) {
        let rd = (self.0 >> 5) & ((1 << 5) - 1);
        let rs = (self.0 >> 10) & ((1 << 5) - 1);
        return (rd as _, rs as _)
    }
    fn to_raw(self) -> u16 { self.0 }
}
#[test]
fn test1() {
    for i in 0 .. 32 {
        for k in 0 .. 32 {
            let cpy = CPY::new(i, k);
            let (urd, urs) = cpy.unpack();
            assert!(i == urd && k == urs);
        }
    }
}
struct DNO(u16);
impl DNO {
    fn new() -> Self { DNO(0) }
    fn to_raw(self) -> u16 { self.0 }
}
#[repr(u8)]
enum GotoExtHeader {
    // register relative
    RelativeOffsetReg, // ip = ip + rn
    // absolute register jump
    AbsoluteAddrReg, // ip = rn
}
/// ip = rn
struct GOTORA(u16);
impl GOTORA {
    fn new(da: u8) -> Self {
        assert!(da < 32);
        let val = Opcode::PCT_GOTO16 as u8 as u16 |
                  ((da as u16) << OPCODE_HEADER1_BITWIDTH) |
                  ((GotoExtHeader::AbsoluteAddrReg as u16) << (OPCODE_HEADER1_BITWIDTH + 5));
        Self(val)
    }
}
/// ip = ip + rn
struct GOTORO(u16);
impl GOTORO {
    fn new(ro: u8) -> Self {
        assert!(ro < 32);
        let val = Opcode::PCT_GOTO16 as u16 |
                  ((ro as u16) << OPCODE_HEADER1_BITWIDTH) |
                  ((GotoExtHeader::RelativeOffsetReg as u16) << (5 + OPCODE_HEADER1_BITWIDTH));
        Self(val)
    }
    fn to_raw(self) -> u16 {
        self.0
    }
}
/// ip = ip + off
struct GOTOIO(u32);
impl GOTOIO {
    fn new(off: i32) -> Self {
        let val = Opcode::GOTO as u32 |
                  ((off as u32) << (OPCODE_HEADER1_BITWIDTH));
        Self(val)
    }
    fn to_raw(self) -> u32 {
        self.0
    }
}
struct AI(u16);
impl AI {
    fn new(rd: u8) -> Self {
        assert!(rd < 32);
        let val = Opcode::COMPL as u16 | ((rd as u16) << OPCODE_HEADER1_BITWIDTH);
        Self(val)
    }
}

struct SDM(u16);
impl SDM {
    fn new(dep_pattern: u8) -> Self {
        let val = Opcode::SDM as u16 | ((dep_pattern as u16) << OPCODE_HEADER1_BITWIDTH);
        Self(val)
    }
    fn to_raw(self) -> u16 {
        self.0
    }
}

fn read_goto_i16_type(insn: u16) -> GotoExtHeader {
    let val = (insn >> (5 + 5)) as u8;
    unsafe { transmute(val) }
}

struct STOP(u16);
impl STOP {
    fn new() -> Self {
        Self(Opcode::HALT as _)
    }
    fn to_raw(self) -> u16 {
        self.0
    }
}

struct ADD(u16);
impl ADD {
    fn new(rd: u8, rs: u8) -> Self {
        assert!(rd < 32);
        assert!(rs < 32);
        let val = Opcode::ADD as u16 |
                  ((rd as u16) << OPCODE_HEADER1_BITWIDTH) |
                  ((rs as u16) << OPCODE_HEADER1_BITWIDTH + 5);
        return Self(val)
    }
    fn to_raw(self) -> u16 {
        self.0
    }
}
struct AND(u16);
impl AND {
    fn new(rd: u8, rs: u8) -> Self {
        assert!(rd < 32);
        assert!(rs < 32);
        let val = Opcode::AND as u16 |
                  ((rd as u16) << OPCODE_HEADER1_BITWIDTH) |
                  ((rs as u16) << OPCODE_HEADER1_BITWIDTH + 5);
        return Self(val)
    }
}
struct OR(u16);
impl OR {
    fn new(rd: u8, rs: u8) -> Self {
        assert!(rd < 32);
        assert!(rs < 32);
        let val = Opcode::AND as u16 |
                  ((rd as u16) << OPCODE_HEADER1_BITWIDTH) |
                  ((rs as u16) << OPCODE_HEADER1_BITWIDTH + 5);
        return Self(val)
    }
}
struct NOT(u16);
impl NOT {
    fn new(rd: u8) -> Self {
        assert!(rd < 32);
        let val = Opcode::AND as u16 |
                  ((rd as u16) << OPCODE_HEADER1_BITWIDTH);
        return Self(val)
    }
}

struct LDS(u16);
impl LDS {
    fn to_raw(self) -> u16 {
        self.0
    }
    fn new(
        rp: u8
    ) -> Self {
        assert!(rp < 32);
        let val = Opcode::LDS as u16 | ((rp as u16) << OPCODE_HEADER1_BITWIDTH);
        return Self(val)
    }
}
struct STS(u16);
impl STS {
    fn to_raw(self) -> u16 {
        self.0
    }
    fn new(
        rp: u8
    ) -> Self {
        assert!(rp < 32);
        let val = Opcode::STS as u16 | ((rp as u16) << OPCODE_HEADER1_BITWIDTH);
        return Self(val)
    }
}
struct LDV(u16);
impl LDV {
    fn to_raw(self) -> u16 {
        self.0
    }
    fn new(
        rd: u8,
        rs: u8
    ) -> Self {
        assert!(rd < 32);
        assert!(rs < 32);
        let val = Opcode::LDV as u16 | ((rd as u16) << 5) | ((rs as u16) << 10);
        Self(val)
    }
}
struct STV(u16);
impl STV {
    fn to_raw(self) -> u16 {
        self.0
    }
    fn new(
        rd: u8,
        rs: u8
    ) -> Self {
        assert!(rd < 32);
        assert!(rs < 32);
        let val = Opcode::STV as u16 | ((rd as u16) << 5) | ((rs as u16) << 10);
        Self(val)
    }
}
struct TC(u16);
impl TC {
    fn new() -> Self {
        Self(Opcode::TC as u16)
    }
    fn to_raw(self) -> u16 {
        self.0
    }
}

#[repr(u8)]
enum ConditionCode {
    Eq, NEq, GT, LT, GE, LE, And, Or
}
struct PCT_IOC(u32);
impl PCT_IOC {
    fn new(
        op1: u8,
        op2: u8,
        cc: ConditionCode,
        imm: i32
    ) -> Self {
        let val =
            Opcode::PCT_GOTOIOC as u32 |
            ((op1 as u32) << 5) |
            ((op2 as u32) << 10) |
            ((cc as u32) << 15) |
            ((imm as u32) << 18);
        Self(val)
    }
    fn to_raw(self) -> u32 {
        self.0
    }
}

#[repr(C)] #[derive(Clone, Copy)]
union Pack {
    int: u64,
    i16x4: [u16;4],
    i16x2_i32x1: (u16,u16,u32),
    i32x2: [u32;2]
}

fn combined_i16x4(
    insns: impl IntoIterator<Item = u16>
) -> u64 {
    let mut items = [0u16;4];
    let mut iter = insns.into_iter();
    for i in 0 .. 4 {
        let item = iter.next();
        match item {
            Some(item) => {
                items[i] = item;
            },
            None => break,
        }
    }
    let items = Pack { i16x4: items };
    return unsafe { items.int };
}
fn combined_i32x2(
    insns: impl IntoIterator<Item = u32>
) -> u64 {
    let mut items = [0u32;2];
    let mut iter = insns.into_iter();
    for i in 0 .. 2 {
        let item = iter.next();
        match item {
            Some(item) => items[i] = item,
            None => break,
        }
    }
    let items = Pack { i32x2: items };
    return unsafe { items.int };
}
fn combined_i16x2(
    insns: impl IntoIterator<Item = u16>
) -> u32 {
    let mut items = [0u16;2];
    let mut iter = insns.into_iter();
    for i in 0 .. 2 {
        let item = iter.next();
        match item {
            Some(item) => items[i] = item,
            None => break,
        }
    }
    let items = unsafe { transmute::<_, u32>(items) };
    return items
}

struct PUState {
    gp_regs: [u32;32],
    ip: u32,
    lmem: *mut u8,
    shmem: *mut u8,
    should_stop: bool
}
impl PUState {
    fn new(
        lmem_size: usize,
        shmem_size: usize,
    ) -> Self {
        let lmem_ptr = unsafe { alloc_zeroed(Layout::from_size_align_unchecked(lmem_size, 4096)) };
        let shmem_ptr = unsafe { alloc_zeroed(Layout::from_size_align_unchecked(shmem_size, 4096)) };
        let st = PUState {
            gp_regs: [0;_],
            lmem: lmem_ptr,
            shmem: shmem_ptr,
            ip: 0,
            should_stop: false
        };
        return st
    }
}

fn run(
    pu_state:&mut PUState,
    code_ptr: *const u64
) {
    loop {
        let ip = pu_state.ip;
        let pack = unsafe { code_ptr.add(ip as usize).read() };
        let pack = Pack { int: pack };
        proc_pack(pu_state, pack);
        if pu_state.should_stop { break }
        pu_state.ip += 1;
    }
}

fn proc_pack(
    pu_state:&mut PUState,
    pack: Pack
) {
    let code_point = &raw const pack;
    let format = compute_pack_format(unsafe { pack.int });
    match format {
        PackFormat::I16x4 => {
            let k@[i1,_,_,_] = unsafe { code_point.cast::<[u16;4]>().read() };
            let oc = read_opcode(&raw const i1 as _);
            if matches!(oc, Opcode::SDM) { // compacted
                let mut iter = decode_compacted(k);
                while let Some(pack) = iter.next() {
                    proc_i16x4(pu_state, unsafe { pack.i16x4 });
                }
            } else {
                proc_i16x4(pu_state, k);
            }
        },
        PackFormat::I16x2_I32x1 => {
            let k@(i1, _, _) = unsafe { code_point.cast::<(u16, u16, u32)>().read() };
            let oc = read_opcode(&raw const i1 as _);
            if matches!(oc, Opcode::SDM) {
                todo!()
            } else {
                proc_i16x2_i32x1(pu_state, k);
            }
        },
        PackFormat::I32x2 => {
            #[allow(unused_variables)]
            let insn = unsafe { code_point.cast::<[u32;2]>().read() };
            proc_i32x2(pu_state, insn);
        },
    }
}
fn read_nth_bit_u16(num:u16, index: usize) -> usize {
    ((num as usize) >> index) & 1
}
fn proc_i32x2(
    pu_state:&mut PUState,
    insn: [u32;2]
) {
    let [i1,i2] = insn;
    proc_goto_i32(pu_state, i2);
    todo!()
}
fn proc_i16x2_i32x1(
    pu_state:&mut PUState,
    insn: (u16, u16, u32)
) {
    let (i1,i2,i3) = insn;
    proc_i16_alu(pu_state,i1);
    proc_i16_alu(pu_state,i2);

    proc_imm_i16(pu_state, i1);
    proc_imm_i16(pu_state, i2);

    proc_imm_i32(pu_state, i3);
    proc_goto_i32(pu_state, i3);

    for i in [i1,i2] {
        let oc = read_opcode(&raw const i as _);
        if matches!(oc, Opcode::HALT) {
            pu_state.should_stop = true;
        }
    }
}
// fn proc_compacted_i16x4(
//     pu_state:&mut PUState,
//     pack: [u16;4]
// ) {
//     let [i1,i2,i3,i4] = pack;
//     let pattern = i1 >> OPCODE_HEADER1_BITWIDTH;
//     let tail_pad = {
//         let tail_bit = !read_nth_bit_u16(pattern, 2) & 1;
//         ((-1i16 as u16) * (tail_bit as u16)) << 3
//     };
//     let pattern = pattern | tail_pad;
//     let num = unsafe { transmute::<_, u64>([i2, i3, i4, 0]) };
//     let mut run = 0u32;
//     loop {
//         let pattern = pattern >> run;
//         let run_ = run;
//         let projection_pattern = {
//             let _0th = 1;
//             let _1st = read_nth_bit_u16(pattern, 0) == read_nth_bit_u16(pattern, 1);
//             let _3rd = read_nth_bit_u16(pattern, 1) == read_nth_bit_u16(pattern, 2);
//             run = run + (_0th as u32) + (_1st as u32) + (_3rd as u32);
//             let c1 = ((_0th as u64) << (1 * 16)) - (_0th as u64);
//             let c2 = ((_1st as u64) << (2 * 16)) - (_1st as u64);
//             let c3 = ((_3rd as u64) << (3 * 16)) - (_3rd as u64);
//             c1 | c2 | c3
//         };
//         let projection_pattern = projection_pattern << (run_ * 16);
//         let pack = (num & projection_pattern) << 16;
//         let pack = unsafe { transmute::<_, [u16;4]>(pack) };
//         proc_i16x4(pu_state, pack);
//         if run >= 3 { break }
//     }
// }
fn decode_compacted(
    pack: [u16;4]
) -> impl Iterator<Item = Pack> {
    core::iter::from_coroutine(#[coroutine] move || {
        let [i1,i2,i3,i4] = pack;
        let pattern = i1 >> OPCODE_HEADER1_BITWIDTH;
        let tail_pad = {
            let tail_bit = !read_nth_bit_u16(pattern, 2) & 1;
            ((-1i16 as u16) * (tail_bit as u16)) << 3
        };
        let pattern = pattern | tail_pad;
        let num = unsafe { transmute::<_, u64>([i2, i3, i4, 0]) };
        let mut run = 0u32;
        loop {
            let pattern = pattern >> run;
            let run_ = run;
            let projection_pattern = {
                let _0th = 1;
                let _1st = read_nth_bit_u16(pattern, 0) == read_nth_bit_u16(pattern, 1);
                let _3rd = read_nth_bit_u16(pattern, 1) == read_nth_bit_u16(pattern, 2);
                run = run + (_0th as u32) + (_1st as u32) + (_3rd as u32);
                let c1 = ((_0th as u64) << (1 * 16)) - (_0th as u64);
                let c2 = ((_1st as u64) << (2 * 16)) - (_1st as u64);
                let c3 = ((_3rd as u64) << (3 * 16)) - (_3rd as u64);
                c1 | c2 | c3
            };
            let projection_pattern = projection_pattern << (run_ * 16);
            let pack = (num & projection_pattern) << 16;
            let pack = Pack { int: pack };
            yield pack;
            if run >= 3 { return }
        }
    })
}

fn proc_i16x4(
    pu_state:&mut PUState,
    pack: [u16;4]
) {
    let k@[i1,i2,i3,i4] = pack;
    proc_i16_alu(pu_state,i1);
    proc_i16_alu(pu_state,i2);
    proc_i16_alu(pu_state,i3);
    proc_i16_alu(pu_state,i4);

    proc_imm_i16(pu_state, i1);
    proc_imm_i16(pu_state, i2);
    proc_imm_i16(pu_state, i3);
    proc_imm_i16(pu_state, i4);

    proc_mem_op(pu_state, i3);
    proc_mem_op(pu_state, i4);

    proc_goto_i16(pu_state, i2);

    for i in k {
        let oc = read_opcode(&raw const i as _);
        if matches!(oc, Opcode::HALT) {
            pu_state.should_stop = true;
        }
    }
}
fn proc_imm_i32(
    pu_state:&mut PUState,
    insn: u32
) {
    let op_code = read_opcode(&raw const insn as _);
    if matches!(op_code, Opcode::PUC32) {
        let rd = (insn << 5) & ((1 << 5) - 1);
        let imm = insn >> 10;
        pu_state.gp_regs[rd as usize] = imm;
    }
}
fn proc_imm_i16(
    pu_state:&mut PUState,
    insn: u16
) {
    let op_code = read_opcode(&raw const insn as _);
    if matches!(op_code, Opcode::CPY) {
        let (rd, rs) = read_i16_alu_op2_args(insn);
        let (rd, rs) = (rd as usize, rs as usize);
        pu_state.gp_regs[rd] = pu_state.gp_regs[rs];
    }
    if matches!(op_code, Opcode::PUC16) {
        let rd = read_i16_alu_op1_arg(insn);
        let val = insn >> 10;
        pu_state.gp_regs[rd as usize] = val as u32;
    }
}
fn proc_goto_i16(
    pu_state:&mut PUState,
    insn: u16
) {
    let op_code = read_opcode(&raw const insn as _);
    if matches!(op_code, Opcode::PCT_GOTO16) {
        let ar = read_i16_alu_op1_arg(insn) as usize;
        let goto_type = read_goto_i16_type(insn);
        match goto_type {
            GotoExtHeader::RelativeOffsetReg => {
                pu_state.ip = pu_state.ip + pu_state.gp_regs[ar] ;
            },
            GotoExtHeader::AbsoluteAddrReg => {
                pu_state.ip = pu_state.gp_regs[ar] ;
            },
        }
    }
}
fn read_goto_i32_imm_offset(insn: u32) -> i32 {
    let high_bit = insn >> 31;
    let left_pad = ((high_bit << 5) - high_bit) << (32 - 5);
    let num = (insn >> OPCODE_HEADER1_BITWIDTH) | left_pad;
    num as i32
}
fn read_gotoioc_imm_offset(insn: u32) -> i32 {
    let imm = read_bit_range_u32(insn, 18..32);
    let high_bit = insn >> 31;
    let pad = 0u32.wrapping_sub(high_bit) << 14;
    let imm = imm | pad;
    imm as i32
}
fn proc_goto_i32(
    pu_state:&mut PUState,
    insn: u32
) {
    let opcode = read_opcode(&raw const insn as _);
    if matches!(opcode, Opcode::GOTO) {
        let offset = read_goto_i32_imm_offset(insn);
        pu_state.ip = pu_state.ip.wrapping_add(offset as u32);
    }
    if matches!(opcode, Opcode::PCT_GOTOIOC) {
        let op1_ri = read_bit_range_u32(insn, 5..10);
        let op2_ri = read_bit_range_u32(insn, 10..15);
        let op1 = pu_state.gp_regs[op1_ri as usize];
        let op2 = pu_state.gp_regs[op2_ri as usize];
        let cc = read_bit_range_u32(insn, 15..18);
        let cc = unsafe { transmute::<_, ConditionCode>(cc as u8) };
        let br = test(cc, op1, op2);
        if br {
            let imm = read_gotoioc_imm_offset(insn);
            pu_state.ip = pu_state.ip.wrapping_add(imm as u32);
        }
    }
}
fn test(cc:ConditionCode, op1: u32, op2:u32) -> bool {
    match cc {
        ConditionCode::Eq => op1 == op2,
        ConditionCode::NEq => op1 != op2,
        ConditionCode::GT => op1 > op2,
        ConditionCode::LT => op1 < op2,
        ConditionCode::GE => op1 >= op2,
        ConditionCode::LE => op1 <= op2,
        ConditionCode::And => (read_bit_range_u32(op1, 0..1) & read_bit_range_u32(op2, 0..1)) == 1,
        ConditionCode::Or => (read_bit_range_u32(op1, 0..1) | read_bit_range_u32(op2, 0..1)) == 1,
    }
}
fn check_should_stop(
    pack: u64
) -> bool {
    match compute_pack_format(pack) {
        PackFormat::I16x4 => {
            let pack = unsafe { transmute::<_, [u16;4]>(pack) };
            for i in pack {
                let oc = read_opcode(&raw const i as _);
                if matches!(oc, Opcode::HALT) {
                    return true
                }
            }
            return false
        },
        PackFormat::I16x2_I32x1 => {
            let (i1,i2, _) = unsafe { transmute::<_, (u16,u16,u32)>(pack) };
            for i in [i1,i2] {
                let oc = read_opcode(&raw const i as _);
                if matches!(oc, Opcode::HALT) {
                    return true
                }
            }
            return false
        },
        PackFormat::I32x2 => {
            return false
        },
    }
}
#[repr(C)]
struct LDSParams {
    rd: u32,
    rs: u32,
    byte_len: u32,
    op_token_rd: u8
}
#[repr(C)]
struct STSParams {
    rd: u32,
    rs: u32,
    byte_len: u32,
    op_token_rd: u8
}

fn proc_mem_op(
    pu_state:&mut PUState,
    insn: u16
) {
    let op_code = read_opcode(&raw const insn as _);
    if matches!(op_code, Opcode::LDS) {
        // write read op to queue
        let arg_addr = read_i16_alu_op1_arg(insn);
        let args_ptr = pu_state.gp_regs[arg_addr as usize] as usize;
        let mop_args_ptr = unsafe { pu_state.lmem.add(args_ptr) };
        let LDSParams { rd, rs, byte_len, op_token_rd } = unsafe { mop_args_ptr.cast::<LDSParams>().read() };
        let src = unsafe { pu_state.shmem.add(rs as usize) };
        let dst = unsafe { pu_state.lmem.add(rd as usize) };
        unsafe { copy_nonoverlapping(src, dst, byte_len as usize) };
        pu_state.gp_regs[op_token_rd as usize] = -1i32 as u32;
    }
    if matches!(op_code, Opcode::STS) {
        // write write op to queue
        let arg_addr = read_i16_alu_op1_arg(insn);
        let args_ptr = pu_state.gp_regs[arg_addr as usize] as usize;
        let mop_args_ptr = unsafe { pu_state.lmem.add(args_ptr) };
        let STSParams { rd, rs, byte_len, op_token_rd } = unsafe { mop_args_ptr.cast::<STSParams>().read() };
        let src = unsafe { pu_state.lmem.add(rs as usize) };
        let dst = unsafe { pu_state.shmem.add(rd as usize) };
        unsafe { copy_nonoverlapping(src, dst, byte_len as usize) };
        pu_state.gp_regs[op_token_rd as usize] = -1i32 as u32;
    }
    if matches!(op_code, Opcode::LDV) {
        let (rd, rs) = read_i16_alu_op2_args(insn);
        let off = pu_state.gp_regs[rs as usize];
        let val = unsafe { pu_state.lmem.add(off as usize).cast::<u32>().read() };
        pu_state.gp_regs[rd as usize] = val;
    }
    if matches!(op_code, Opcode::STV) {
        let (rd, rs) = read_i16_alu_op2_args(insn);
        let off = pu_state.gp_regs[rd as usize];
        let val = pu_state.gp_regs[rs as usize];
        unsafe { pu_state.lmem.add(off as usize).cast::<u32>().write(val) };
    }
}
fn proc_i16_alu(pu_state:&mut PUState, insn: u16) {
    let op_code = read_opcode(&raw const insn as _);
    if matches!(op_code, Opcode::ADD) {
        let (rd, rs) = read_i16_alu_op2_args(insn);
        let (rd, rs) = (rd as usize, rs as usize);
        let val1 = pu_state.gp_regs[rd];
        let val2 = pu_state.gp_regs[rs];
        pu_state.gp_regs[rd] = val1 + val2 ;
    }
    if matches!(op_code, Opcode::COMPL) {
        let rd = read_i16_alu_op1_arg(insn) as usize;
        pu_state.gp_regs[rd] = -(pu_state.gp_regs[rd] as i32) as u32;
    }
    if matches!(op_code, Opcode::AND) {
        let (rd, rs) = read_i16_alu_op2_args(insn);
        let (rd, rs) = (rd as usize, rs as usize);
        let val1 = pu_state.gp_regs[rd];
        let val2 = pu_state.gp_regs[rs];
        pu_state.gp_regs[rd] = val1 & val2 ;
    }
    if matches!(op_code, Opcode::OR) {
        let (rd, rs) = read_i16_alu_op2_args(insn);
        let (rd, rs) = (rd as usize, rs as usize);
        let val1 = pu_state.gp_regs[rd];
        let val2 = pu_state.gp_regs[rs];
        pu_state.gp_regs[rd] = val1 | val2 ;
    }
    if matches!(op_code, Opcode::NOT) {
        let rd = read_i16_alu_op1_arg(insn) as usize;
        pu_state.gp_regs[rd] = !pu_state.gp_regs[rd];
    }
}

#[test] #[ignore]
fn offset_goto_is_off_by_one() {
    let mut st = PUState::new(128, 0);
    let code = [
        combined_i16x4([
            PUCS::new(0, 2 - 1).to_raw(),
        ]),
        combined_i16x4([ // A + 0
            DNO::new().to_raw(),
            GOTORO::new(0).to_raw()
        ]),
        combined_i16x4([ // A + 1
            PUCS::new(1, 8).to_raw(),
            PUCS::new(2, 9).to_raw(),
            PUCS::new(3, 10).to_raw(),
            STOP::new().to_raw()
        ]),
        combined_i16x4([ // A + 2
            PUCS::new(1, 47).to_raw(),
            PUCS::new(2, 46).to_raw(),
            PUCS::new(3, 45).to_raw(),
            PUCS::new(4, 44).to_raw(),
        ]),
        combined_i16x4([
            STOP::new().to_raw()
        ])
    ] as [u64;_];
    let code_ptr = (&raw const code) as *const u64;
    run(&mut st, code_ptr);
    println!("{:#?}", st.gp_regs);
}

#[test]
fn jumper() {
    let mut st = PUState::new(128, 0);
    let code = [
        combined_i32x2([ // A - 2
            combined_i16x2([
                DNO::new().to_raw(),
                DNO::new().to_raw()
            ]),
            GOTOIO::new(2-1).to_raw()
        ]),
        combined_i16x4([ // A - 1
            PUCS::new(1, 8).to_raw(),
            PUCS::new(2, 9).to_raw(),
            PUCS::new(3, 10).to_raw(),
            STOP::new().to_raw(),
        ]),
        combined_i32x2([ // A + 0
            combined_i16x2([
                DNO::new().to_raw(),
                DNO::new().to_raw()
            ]),
            GOTOIO::new(-1-1).to_raw()
        ]),
    ] as [u64;_];
    let code_ptr = (&raw const code) as *const u64;
    run(&mut st, code_ptr);
    println!("{:#?}", st.gp_regs);
}

#[test]
fn memcpy_out_in() {
    let mut st = PUState::new(128, 512);
    let msg = "privet kak dela";
    unsafe { copy_nonoverlapping(msg.as_ptr(), st.shmem.add(128), msg.len()) };
    unsafe { st.lmem.add(64).cast::<LDSParams>().write(LDSParams { rd: 0, rs: 128, byte_len: msg.len() as u32, op_token_rd: 1 }); };
    let code = [
        combined_i32x2([
            combined_i16x2([
                DNO::new().to_raw(),
                DNO::new().to_raw(),
            ]),
            PUCE::new(0, 64).to_raw(),
        ]),
        combined_i16x4([
            DNO::new().to_raw(),
            DNO::new().to_raw(),
            STOP::new().to_raw(),
            LDS::new(0).to_raw()
        ])
    ] as [u64;_];
    let code_ptr = (&raw const code) as *const u64;
    run(&mut st, code_ptr);
    let msg_ = unsafe { core::str::from_raw_parts(st.lmem, msg.len()) };
    assert!(msg == msg_);
}
#[test]
fn memcpy_in_out() {
    let mut st = PUState::new(128, 512);
    let msg = "privet kak dela";
    unsafe { copy_nonoverlapping(msg.as_ptr(), st.lmem.add(128), msg.len()) };
    unsafe { st.lmem.add(64).cast::<STSParams>().write(STSParams { rd: 0, rs: 128, byte_len: msg.len() as u32, op_token_rd: 1 }); };
    let code = [
        combined_i32x2([
            combined_i16x2([
                DNO::new().to_raw(),
                DNO::new().to_raw(),
            ]),
            PUCE::new(0, 64).to_raw(),
        ]),
        combined_i16x4([
            DNO::new().to_raw(),
            DNO::new().to_raw(),
            STOP::new().to_raw(),
            STS::new(0).to_raw()
        ])
    ] as [u64;_];
    let code_ptr = (&raw const code) as *const u64;
    run(&mut st, code_ptr);
    let msg_ = unsafe { core::str::from_raw_parts(st.shmem, msg.len()) };
    assert!(msg == msg_);
}
#[test]
fn memcpy_in_out_local() {
    let mut st = PUState::new(128, 0);
    let st_val = 7;
    let code = [
        combined_i16x4([
            SDM::new(0b0000_0100).to_raw(),
            PUCS::new(0, st_val).to_raw(),
            PUCS::new(1, 8).to_raw(),
            STV::new(1, 0).to_raw()
        ]),
        combined_i16x4([
            DNO::new().to_raw(),
            DNO::new().to_raw(),
            STOP::new().to_raw(),
            LDV::new(2, 1).to_raw(),
        ])
    ] as [u64;_];
    let code_ptr = (&raw const code) as *const u64;
    run(&mut st, code_ptr);
    assert!(st.gp_regs[2] == st_val as u32);
}

fn gen_fibs<const N:usize>() -> [u32;N] {
    let mut vals = [0;N];
    let (mut a, mut b) = (1,1);
    for i in 0 .. N as _ {
        let tmp = b + a;
        b = a;
        a = tmp;
        vals[i] = b;
    }
    return vals;
}

#[test] #[ignore]
fn fib() {
    let mut st = PUState::new(128, 0);
    const ITEM_COUNT: usize = 10;
    let code = [
        combined_i32x2([
            combined_i16x2([
                PUCS::new(1, 4).to_raw(), // index bump amount
                PUCS::new(2, 0).to_raw(), // current addr
            ]),
            PUCE::new(0, (ITEM_COUNT * 4) as _).to_raw() // end addr
        ]),
        combined_i16x4([
            PUCS::new(3, 1).to_raw(), // a
            PUCS::new(4, 1).to_raw(), // b
        ]),
        // loop start
        combined_i16x4([
            DNO::new().to_raw(),
            DNO::new().to_raw(),
            STV::new(2, 3).to_raw(),
        ]),
        combined_i16x4([
            SDM::new(0b0000_0010).to_raw(),
            CPY::new(5, 4).to_raw(),
            ADD::new(5, 3).to_raw(),
            CPY::new(4, 3).to_raw(),
        ]),
        combined_i16x4([
            CPY::new(3, 5).to_raw(),
            ADD::new(2, 1).to_raw(),
        ]),
        combined_i32x2([
            combined_i16x2([]),
            PCT_IOC::new(2, 0, ConditionCode::NEq, -3-1).to_raw(),
        ]),
        combined_i16x4([
            STOP::new().to_raw(),
        ])
    ];
    let code_ptr = (&raw const code) as *const u64;
    run(&mut st, code_ptr);
    let computed = unsafe { core::slice::from_raw_parts(st.lmem.cast::<u32>(), ITEM_COUNT as usize) };
    let ground_truth = gen_fibs::<ITEM_COUNT>();
    assert!(computed == ground_truth);
}



#[test] #[ignore]
fn compact_pack_decode() {

    // let mut st = PUState::new(128, 0);
    // let code = [
    //     combined_i16x4([
    //         POLP::new(0b0000_0100).to_raw(),
    //         PUCS::new(0, 17).to_raw(),
    //         PUCS::new(1, 33).to_raw(),
    //         HALT::new().to_raw(),
    //     ])
    // ] as [u64;_];
    // let code_ptr = (&raw const code) as *const u64;
    // run(&mut st, code_ptr);
    // println!("{:#?}", st.gp_regs);

    // let mut st = PUState::new(128, 0);
    // let code = [
    //     combined_i16x4([
    //         POLP::new(0b0000_0101).to_raw(),
    //         PUCS::new(0, 17).to_raw(),
    //         PUCS::new(1, 33).to_raw(),
    //         HALT::new().to_raw(),
    //     ])
    // ] as [u64;_];
    // let code_ptr = (&raw const code) as *const u64;
    // run(&mut st, code_ptr);
    // println!("{:#?}", st.gp_regs);

    // let mut st = PUState::new(128, 0);
    // let code = [
    //     combined_i16x4([
    //         PSP::new(0b0000_0111).to_raw(),
    //         PUCS::new(0, 17).to_raw(),
    //         PUCS::new(1, 33).to_raw(),
    //         STOP::new().to_raw(),
    //     ])
    // ] as [u64;_];
    // let code_ptr = (&raw const code) as *const u64;
    // run(&mut st, code_ptr);
    // println!("{:#?}", st.gp_regs);

    // let mut st = PUState::new(128, 0);
    // let code = [
    //     combined_i16x4([
    //         POLP::new(0b0000_0010).to_raw(),
    //         PUCS::new(0, 17).to_raw(),
    //         PUCS::new(1, 33).to_raw(),
    //         HALT::new().to_raw(),
    //     ])
    // ] as [u64;_];
    // let code_ptr = (&raw const code) as *const u64;
    // run(&mut st, code_ptr);
    // println!("{:#?}", st.gp_regs);


    let mut st = PUState::new(128, 0);
    let code = [
        combined_i16x4([
            SDM::new(0b0000_0101).to_raw(),
            PUCS::new(0, 17).to_raw(),
            PUCS::new(1, 33).to_raw(),
        ]),
        combined_i16x4([
            STOP::new().to_raw(),
        ])
    ] as [u64;_];
    let code_ptr = (&raw const code) as *const u64;
    run(&mut st, code_ptr);
    println!("{:#?}", st.gp_regs);

}

