use core::{alloc::Layout, hint::unreachable_unchecked, mem::{transmute, ManuallyDrop, MaybeUninit}, ops::Range, ptr::{copy_nonoverlapping, null_mut}, sync::atomic::{fence, AtomicBool, AtomicU32, Ordering}};
use std::{alloc::alloc_zeroed, simd::{cmp::SimdPartialEq, num::SimdUint}};
use core::simd::Simd;
use atomic_spsc_queue::RingQueue;

use crate::*;

type IFetchBlock = [u64;8];
#[repr(C)] #[derive(Clone, Copy)]
union AddrData {
    simd: Simd<u32, 8>,
    items: [u32;8],
}
#[repr(C)] #[derive(Clone, Copy)]
union RelevanceData {
    simd: Simd<u8, 8>,
    items: [u8;8],
}
struct ICache {
    addr_bits: [AddrData; Self::ENTRY_COUNT],
    relevance_counters: [RelevanceData;Self::ENTRY_COUNT],
    occupation_map: [u8;Self::ENTRY_COUNT],
    mem: [[IFetchBlock;8]; Self::ENTRY_COUNT],
}
impl ICache {
    const ENTRY_COUNT: usize = 32;
}
static_assert!(ICache::ENTRY_COUNT.is_power_of_two());

macro static_assert($cond:expr) {
    const _ : () = if !$cond { panic!() } else { () };
}

fn new_icache() -> ICache {
    unsafe { MaybeUninit::<ICache>::zeroed().assume_init() }
}

fn bucket_index_from_addr(addr: u32) -> usize {
    (addr as usize) & (ICache::ENTRY_COUNT - 1)
}
fn insert(
    icache:&mut ICache,
    addr: u32,
    line_data: *const IFetchBlock
) {
    let bucket_index = bucket_index_from_addr(addr);
    let occupation_map = &mut icache.occupation_map[bucket_index as usize];
    let rel_ctrs = &mut icache.relevance_counters[bucket_index as usize];
    let no_free_space = *occupation_map == u8::MAX;
    let vacant_slot_index = if no_free_space {
        // we have to evict the item with lowest relevance count,
        let simd_target = unsafe{&rel_ctrs.simd};
        let least_relevant = simd_target.reduce_min();
        let least_relevant_ = simd_target.simd_eq(Simd::splat(least_relevant));
        let least_relevant_ = least_relevant_.to_bitmask();
        let least_relevant_index = least_relevant_.trailing_zeros();
        least_relevant_index
    } else {
        let vacant_slot = occupation_map.trailing_ones();
        vacant_slot
    };
    let dst = &raw mut icache.mem[bucket_index as usize][vacant_slot_index as usize];
    unsafe { copy_nonoverlapping(line_data, dst, 1) };
    unsafe{icache.addr_bits[bucket_index as usize].items[vacant_slot_index as usize] = addr};
    let fresh_insertion_relevance_value = u8::MAX >> 1; // half of max relevance
    unsafe {rel_ctrs.items[vacant_slot_index as usize] = fresh_insertion_relevance_value};
    *occupation_map |= 1 << vacant_slot_index;
}

fn decay_relevance(
    icache:&mut ICache
) {
    let decay_rate = 1;
    for counter_block in &mut icache.relevance_counters {
        unsafe {
            counter_block.simd =
            counter_block.simd.saturating_sub(Simd::splat(decay_rate));
        }
    }
}

fn lookup(
    icache: &mut ICache,
    address: u32,
) -> Option<*const IFetchBlock> {
    let bucket_index = bucket_index_from_addr(address);
    let mask = unsafe{icache.addr_bits[bucket_index].simd.simd_eq(Simd::splat(address))};
    let mask = mask.to_bitmask();
    let nothing = mask == 0 || mask == u8::MAX as u64;
    if nothing {
        return None
    }
    debug_assert!(mask.is_power_of_two(), "shadowing detected");
    let index = mask.trailing_zeros();
    let ptr = &icache.mem[bucket_index][index as usize];
    let relevance_boost = 8;
    unsafe {
        let relevance_counter = &mut icache.relevance_counters[bucket_index].items[index as usize];
        *relevance_counter = relevance_counter.saturating_add(relevance_boost)
    };
    return Some(ptr)
}


#[test]
fn insert_basics() {
    let mut icache = new_icache();
    for i in 0 .. 8 {
        insert(&mut icache, 32*i, &[i as u64;_]);
    }
    // insert(&mut icache, 32*9, &[69;_]);
    let outcome = lookup(&mut icache, 32);
    match outcome {
        Some(ptr) => {
            println!("{:?}", unsafe{*ptr})
        },
        None => {

        },
    }
}
#[repr(u8)]
enum SchedType {
    None, Offset, Addr
}
#[repr(C)]
union ShadowIP {
    u32: u32,
    i32: i32,
    atomic: ManuallyDrop<AtomicU32>
}
struct Frontend {
    icache: ICache,
    decode_sink: RingQueue<Pack>,
    ip: u32,
    shadow_ip: AtomicU32,
    tc_sched: SchedType,
    tc_resolved: AtomicBool
}

fn new_frontend() -> Frontend {
    Frontend {
        icache: new_icache(),
        decode_sink: RingQueue::new(32),
        ip: 0,
        shadow_ip: AtomicU32::new(0),
        tc_sched: SchedType::None,
        tc_resolved: AtomicBool::new(false)
    }
}

struct CacheFillStation(MaybeUninit<IFetchBlock>, u32, AtomicU32);

fn frontend_logic(shmem_ptr:*mut u8, lmem_ptr:*mut u8) {
    let mut fe = new_frontend();
    let mut fill_station = CacheFillStation(MaybeUninit::uninit(), 0, AtomicU32::new(0));
    let everything_should_stop = AtomicBool::new(false);
    let exe = unsafe {
        let sink = &fe.decode_sink;
        let shadow_ip = &fe.shadow_ip;
        let cf = &fe.tc_resolved;
        let should_stop_everything = &everything_should_stop;
        let lmem_ptr = lmem_ptr as usize;
        let shmem_ptr = shmem_ptr as usize;
        std::thread::Builder::new().spawn_unchecked(move || {
            let lmem_ptr = lmem_ptr as *mut u8;
            let shmem_ptr = shmem_ptr as *mut u8;
            execution_logic(sink, shadow_ip, cf, &should_stop_everything, lmem_ptr, shmem_ptr);
        }).unwrap()
    };
    let mem_server = unsafe {
        let fill_station = &raw mut fill_station;
        let fill_station = fill_station as usize;
        let mem_ptr = shmem_ptr;
        let mem_ptr = mem_ptr as usize;
        let should_stop_everything = &everything_should_stop;
        std::thread::Builder::new().spawn_unchecked(move || {
            let mem_ptr = mem_ptr as *mut u8;
            let fill_station = fill_station as *mut CacheFillStation;
            let fill_station = &mut *fill_station;
            loop {
                if should_stop_everything.load(Ordering::Relaxed) { return }
                let should_serve_cache_fill = fill_station.2.load(Ordering::Acquire) == 1;
                if should_serve_cache_fill {
                    let cl = &raw mut fill_station.0;
                    let addr = (fill_station.1 as usize) << 3;
                    let ptr = mem_ptr.add(addr);
                    copy_nonoverlapping(ptr, cl.cast(), size_of::<IFetchBlock>());
                    fill_station.2.store(2, Ordering::Release);
                }
                //
            }
        }).unwrap()
    };
    let mut cl = MaybeUninit::<IFetchBlock>::uninit();
    let mut is_loaded = false;
    let mut boundry_addr = 0;
    'main:loop {
        let cl_aligned_boundry = fe.ip & !((size_of::<IFetchBlock>() - 1) as u32);
        let available_locally = cl_aligned_boundry == boundry_addr;
        if !available_locally || !is_loaded {
            let cl_ = lookup(&mut fe.icache, fe.ip);
            let cl_ = match cl_ {
                Some(cl) => unsafe{*cl},
                None => {
                    // stall until the target is delivered
                    fill_station.1 = fe.ip;
                    fill_station.2.store(1, Ordering::Release);
                    while fill_station.2.load(Ordering::Relaxed) != 2 {}
                    fence(Ordering::SeqCst);
                    unsafe { fill_station.0.assume_init_read() }
                },
            };
            cl.write(cl_);
            is_loaded = true;
            boundry_addr = cl_aligned_boundry;
        }
        let read_offset = fe.ip & ((size_of::<IFetchBlock>() - 1) as u32);
        let read_index = read_offset >> 3;
        let pack = unsafe { cl.assume_init_ref()[read_index as usize] };
        let pack = Pack { int: pack };
        let k@[i1,_,i3,i4] = unsafe { pack.i16x4 };
        // check if the tail instruction is a direct control transition
        let i3_opcode = read_opcode(&raw const i3 as _);
        let is_goto = matches!(i3_opcode, Opcode::GOTO);
        if is_goto {
            // it is a direct control transition
            let offset = unsafe { read_goto_i32_imm_offset(pack.i16x2_i32x1.2) };
            fe.ip = fe.ip.wrapping_add(offset as u32); // short unconditional jump
        }
        let i4_opcode = read_opcode(&raw const i4 as _);
        // check if the tail instruction is a control transfer instrucion
        let is_control_transfer = matches!(i4_opcode, Opcode::TC) && !matches!(fe.tc_sched, SchedType::None);
        if is_control_transfer {
            // it is a control transfer instrucion
            // its time to jump and theres a resolution possibly pending.
            // if its not ready, stall!
            while !fe.tc_resolved.load(Ordering::Relaxed) {}
            fence(Ordering::SeqCst);
            let val = fe.shadow_ip.load(Ordering::Relaxed);
            match fe.tc_sched {
                SchedType::Offset => fe.ip = fe.ip.wrapping_add(val),
                SchedType::Addr => fe.ip = val,
                SchedType::None => unsafe { unreachable_unchecked() },
            }
        }
        // check if the last instruction is a control transition preparation instruction
        let is_pct_16 = matches!(i4_opcode, Opcode::PCT_GOTO16);
        if is_pct_16 {
            let ext = unsafe { transmute::<_, GotoExtHeader>((i4 >> 10) as u8) };
            match ext {
                GotoExtHeader::RelativeOffsetReg => {
                    fe.tc_sched = SchedType::Offset;
                },
                GotoExtHeader::AbsoluteAddrReg => {
                    fe.tc_sched = SchedType::Addr;
                },
            }
            // we dont know the target at this point in the pipeline
            fe.tc_resolved.store(false, Ordering::Release);
        }
        let is_pct_32 = matches!(i3_opcode, Opcode::PCT_GOTOIOC);
        if is_pct_32 {
            fe.tc_sched = SchedType::Offset;
            let comb = unsafe { pack.i16x2_i32x1.2 };
            let offset = read_gotoioc_imm_offset(comb);
            let offset = offset << 3;
            fe.shadow_ip.store(offset as u32, Ordering::Relaxed);
            fe.tc_resolved.store(false, Ordering::Release);
        }
        let opcode = read_opcode(&raw const i1 as _);
        let needs_unpack = matches!(opcode, Opcode::SDM);
        if needs_unpack {
            let mut iter = decode_compacted(k);
            while let Some(pack) = iter.next() {
                // stall if backend cant catch up
                let pack = MaybeUninit::new(pack);
                while !fe.decode_sink.enqueue_item(&pack) {}
            }
        } else {
            // no need for unpacking
            // stall if backend cant catch up
            let pack = MaybeUninit::new(pack);
            while !fe.decode_sink.enqueue_item(&pack) {}
        }
        let should_transition_control = is_goto || is_control_transfer;
        decay_relevance(&mut fe.icache);
        if !is_goto && !is_control_transfer {
            fe.ip += 8;
        }
        let should_stop = check_should_stop(unsafe { pack.int });
        if should_stop {
            everything_should_stop.store(true, Ordering::Relaxed);
            mem_server.join().unwrap();
            exe.join().unwrap();
            return
        }
        if should_transition_control {
            continue 'main;
        }
    }
}
struct PUState2 {
    gp_regs: [u32;32],
    lmem: *mut u8,
    shmem: *mut u8,
    shadow_ip: *const AtomicU32,
    completion_flag: *const AtomicBool,
}
impl PUState2 {
    fn new(
        lmem_ptr: *mut u8,
        shmem_ptr:*mut u8,
        shadow_ip: *const AtomicU32,
        completion_flag: *const AtomicBool
    ) -> Self {
        let st = PUState2 {
            gp_regs: [0;_],
            lmem: lmem_ptr,
            shadow_ip,
            completion_flag,
            shmem: shmem_ptr
        };
        return st
    }
}
fn proc_i16_alu_exe(pu_state:&mut PUState2, insn: u16) {
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
fn proc_goto_i16_exe_side(
    pu_state:&mut PUState2,
    insn: u16
) {
    let op_code = read_opcode(&raw const insn as _);
    if matches!(op_code, Opcode::PCT_GOTO16) {
        let ar = read_i16_alu_op1_arg(insn) as usize;
        let val = pu_state.gp_regs[ar];
        unsafe {
            (*pu_state.shadow_ip).store(val, Ordering::Relaxed);
            (*pu_state.completion_flag).store(true, Ordering::Release);
        };
    }
}
fn proc_imm_i16_exe(
    pu_state:&mut PUState2,
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
fn proc_mem_op_exe(
    pu_state:&mut PUState2,
    insn: u16,
) {
    let op_code = read_opcode(&raw const insn as _);
    if matches!(op_code, Opcode::LDS) {
        // write read op to queue
        let arg_addr = read_i16_alu_op1_arg(insn);
        let args_ptr = pu_state.gp_regs[arg_addr as usize] as usize;
        let mop_args_ptr = unsafe { pu_state.lmem.add(args_ptr) };
        // todo: more faithful repr???
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
fn proc_i16x4_2(
    pu_state:&mut PUState2,
    pack: [u16;4],
) {
    let [i1,i2,i3,i4] = pack;
    proc_i16_alu_exe(pu_state,i1);
    proc_i16_alu_exe(pu_state,i2);
    proc_i16_alu_exe(pu_state,i3);
    proc_i16_alu_exe(pu_state,i4);

    proc_imm_i16_exe(pu_state, i1);
    proc_imm_i16_exe(pu_state, i2);
    proc_imm_i16_exe(pu_state, i3);
    proc_imm_i16_exe(pu_state, i4);

    proc_mem_op_exe(pu_state, i3);
    proc_mem_op_exe(pu_state, i4);

    proc_goto_i16_exe_side(pu_state, i2);
}
fn proc_imm_i32_exe(
    pu_state:&mut PUState2,
    insn: u32
) {
    let op_code = read_opcode(&raw const insn as _);
    if matches!(op_code, Opcode::PUC32) {
        let rd = (insn << 5) & ((1 << 5) - 1);
        let imm = insn >> 10;
        pu_state.gp_regs[rd as usize] = imm;
    }
}
fn proc_goto_i32_exe(
    pu_state:&mut PUState2,
    insn: u32
) {
    let opcode = read_opcode(&raw const insn as _);
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
            let off = imm << 3;
            unsafe {
                (*pu_state.shadow_ip).store(off as u32, Ordering::Relaxed);
                (*pu_state.completion_flag).store(true, Ordering::Release);
            };
        } else {
            unsafe {
                (*pu_state.shadow_ip).store(8, Ordering::Relaxed);
                (*pu_state.completion_flag).store(true, Ordering::Release);
            };
        }
    }
}
fn proc_i16x2_i32x1_exe(
    pu_state:&mut PUState2,
    insn: (u16, u16, u32)
) {
    let (i1,i2,i3) = insn;
    proc_i16_alu_exe(pu_state,i1);
    proc_i16_alu_exe(pu_state,i2);

    proc_imm_i16_exe(pu_state, i1);
    proc_imm_i16_exe(pu_state, i2);

    proc_imm_i32_exe(pu_state, i3);
    proc_goto_i32_exe(pu_state, i3);
}
fn proc_i32x2_exe(
    pu_state:&mut PUState2,
    insn: [u32;2]
) {
    let [i1,i2] = insn;
    proc_goto_i32_exe(pu_state, i2);
    todo!()
}
fn proc_pack_2(
    pu_state:&mut PUState2,
    pack: Pack
) {
    let format = compute_pack_format(unsafe { pack.int });
    match format {
        PackFormat::I16x4 => {
            let k@[i1,_,_,_] = unsafe { pack.i16x4 };
            let oc = read_opcode(&raw const i1 as _);
            if matches!(oc, Opcode::SDM) { // compacted
                let mut iter = decode_compacted(k);
                while let Some(pack) = iter.next() {
                    proc_i16x4_2(pu_state, unsafe { pack.i16x4 });
                }
            } else {
                proc_i16x4_2(pu_state, k);
            }
        },
        PackFormat::I16x2_I32x1 => {
            let k@(i1, _, _) = unsafe { pack.i16x2_i32x1 };
            let oc = read_opcode(&raw const i1 as _);
            if matches!(oc, Opcode::SDM) {
                todo!()
            } else {
                proc_i16x2_i32x1_exe(pu_state, k);
            }
        },
        PackFormat::I32x2 => {
            #[allow(unused_variables)]
            let insn = unsafe { pack.i32x2 };
            proc_i32x2_exe(pu_state, insn);
        },
    }
}
fn execution_logic(
    decode_sink_ref: &RingQueue<Pack>,
    shadow_ip: *const AtomicU32,
    completion_flag: *const AtomicBool,
    stop_signal: &AtomicBool,
    lmem_ptr:*mut u8,
    shmem_ptr:*mut u8
) {
    let mut state = PUState2::new(lmem_ptr, shmem_ptr, shadow_ip, completion_flag);
    let mut item = MaybeUninit::uninit();
    loop {
        if stop_signal.load(Ordering::Relaxed) {
            return
        }
        let ok = decode_sink_ref.dequeue_item(&mut item);
        if !ok { continue }
        fence(Ordering::SeqCst);
        let pack = unsafe { item.assume_init_read() };
        proc_pack_2(&mut state, pack);
    }
}


#[test]
fn fl_test() {
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
            PCT_IOC::new(2, 0, ConditionCode::NEq, -4).to_raw(),
        ]),
        combined_i16x4([
            DNO::new().to_raw(),
            DNO::new().to_raw(),
            DNO::new().to_raw(),
            TC::new().to_raw()
        ]),
        combined_i16x4([
            STOP::new().to_raw(),
        ])
    ];
    let code_ptr = (&raw const code) as *const u8;
    let shmem = unsafe { std::alloc::alloc(Layout::from_size_align_unchecked(4096, 4096)) };
    let size = size_of_val(&code);
    unsafe { copy_nonoverlapping(code_ptr, shmem, size) };
    let lmem = unsafe { alloc_zeroed(Layout::from_size_align_unchecked(4096, 4096)) };

    frontend_logic(shmem, lmem);

    let computed = unsafe { core::slice::from_raw_parts(lmem.cast::<u32>(), ITEM_COUNT as usize) };
    println!("{:?}", computed);
    let ground_truth = gen_fibs::<ITEM_COUNT>();
    assert!(computed == ground_truth);
}