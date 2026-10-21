"""Run fw_m4.elf on an emulated ARM Cortex-M4 (Unicorn engine, QEMU-derived) and count executed instructions.
Semihosting calls (bkpt 0xAB) are serviced in Python: SYS_WRITE0 prints, SYS_EXIT stops.
Usage: python emulate_m4.py fw_m4.elf
"""
import sys, struct
from unicorn import *
from unicorn.arm_const import *
from elftools.elf.elffile import ELFFile

elf = ELFFile(open(sys.argv[1], "rb"))
uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB | UC_MODE_MCLASS)
uc.ctl_set_cpu_model(UC_CPU_ARM_CORTEX_M4)
uc.mem_map(0x00000000, 4 * 1024 * 1024)      # flash
uc.mem_map(0x20000000, 4 * 1024 * 1024)      # ram
uc.mem_map(0xE000E000, 0x1000)               # system control space (CPACR write target)
for seg in elf.iter_segments():
    if seg["p_type"] == "PT_LOAD" and seg["p_filesz"] > 0:
        uc.mem_write(seg["p_paddr"], seg.data()[: seg["p_filesz"]])
vec = uc.mem_read(0, 8); sp, reset = struct.unpack("<II", bytes(vec))
try:
    uc.reg_write(UC_ARM_REG_CPACR, 0x00F00000)   # enable CP10/CP11 (FPU)
except Exception as e:
    print("CPACR write via register failed:", e)
uc.reg_write(UC_ARM_REG_SP, sp)
count = {"insns": 0, "in_encode": 0}
# locate codec_encode symbol range to count instructions inside the codec only
syms = {s.name: (s["st_value"] & ~1, s["st_size"]) for s in elf.get_section_by_name(".symtab").iter_symbols() if s["st_info"]["type"] == "STT_FUNC"}
enc_lo, enc_sz = syms["codec_encode"]; enc_hi = enc_lo + enc_sz
state = {"depth": 0}
def hook_code(uc, address, size, user):
    count["insns"] += 1
    # count everything executed between entry to codec_encode and its return (callees included):
    # track via LR: when PC == entry, remember LR; when PC == that LR, leave.
    if address == enc_lo:
        state["ret"] = uc.reg_read(UC_ARM_REG_LR) & ~1; state["in"] = True
    if state.get("in"):
        count["in_encode"] += 1
        if address == state.get("ret"):
            state["in"] = False; count["in_encode"] -= 1
uc.hook_add(UC_HOOK_CODE, hook_code)
out = []
def hook_intr(uc, intno, user):
    r0 = uc.reg_read(UC_ARM_REG_R0); r1 = uc.reg_read(UC_ARM_REG_R1)
    if r0 == 0x04:   # SYS_WRITE0
        s = b""; a = r1
        while True:
            c = bytes(uc.mem_read(a, 1)); a += 1
            if c == b"\0": break
            s += c
        out.append(s.decode()); print(s.decode(), end="", flush=True)
        pc = uc.reg_read(UC_ARM_REG_PC) & ~1
        if int.from_bytes(bytes(uc.mem_read(pc, 2)), "little") & 0xFF00 == 0xBE00:
            pc += 2                                   # skip the bkpt instruction itself
        uc.reg_write(UC_ARM_REG_PC, pc | 1)
    elif r0 == 0x18: # SYS_EXIT
        uc.emu_stop()
    else:
        print("unhandled semihosting op", hex(r0)); uc.emu_stop()
uc.hook_add(UC_HOOK_INTR, hook_intr)
try:
    import time; t=time.time(); uc.emu_start(reset | 1, 0xFFFFFFFF, count=int(__import__("os").environ.get("MAXINSN", "0")))
    print("elapsed", round(time.time()-t,1), "s; final pc", hex(uc.reg_read(UC_ARM_REG_PC)))
except UcError as e:
    print("emulation error:", e, "pc=", hex(uc.reg_read(UC_ARM_REG_PC)))
print(f"semihosting lines: {len(out)}")
print(f"INSTRUCTIONS total={count['insns']} inside_codec_encode={count['in_encode']}")
