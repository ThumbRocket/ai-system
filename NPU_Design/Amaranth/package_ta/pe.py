from amaranth import *
from mac import MAC


class PE(Elaboratable):
    def __init__(self, num_bits, acc_bits, cnt_bits, signed=True):
        self.num_bits = num_bits
        self.acc_bits = acc_bits
        self.cnt_bits = cnt_bits
        self.signed = signed

        assert acc_bits >= 2 * num_bits

        self.in_init = Signal(cnt_bits)
        self.in_rst = Signal(1, reset_less=True)

        self.in_a = Signal(Shape(num_bits, signed=signed))
        self.in_b = Signal(Shape(num_bits, signed=signed))

        self.out_d = Signal(Shape(acc_bits, signed=signed))
        self.out_d_valid = Signal(1)
        self.out_ovf = Signal(1)

        self.cnt_target = Signal(cnt_bits)
        self.cnt = Signal(cnt_bits)
        self.cnt_ovf = Signal(1)
        self.cnt_next = Signal(cnt_bits + 1)

        self.mac = MAC(num_bits=num_bits, acc_bits=acc_bits, signed=signed)
        self.is_exec = Signal(1)

    def elaborate(self, platform):
        m = Module()

        m.submodules.mac = mac = ResetInserter(self.mac.in_rst)(self.mac)

        m.d.comb += [
            mac.in_a.eq(self.in_a),
            mac.in_a_valid.eq(self.is_exec),
            mac.in_b.eq(self.in_b),
            mac.in_b_valid.eq(self.is_exec),
            mac.in_rst.eq(~self.is_exec & self.in_init.any() | self.in_rst),
            self.out_d.eq(mac.out_d),
            self.out_d_valid.eq(mac.out_d_valid & ~self.is_exec),
            self.out_ovf.eq(mac.out_ovf),
            self.cnt_next.eq(self.cnt + 1),
        ]

        with m.FSM(reset="INIT"):
            with m.State("INIT"):
                with m.If(self.in_init):
                    m.d.sync += [
                        self.cnt_target.eq(self.in_init),
                        self.cnt.eq(0),
                        self.cnt_ovf.eq(0),
                        self.is_exec.eq(1),
                    ]
                    m.next = "EXEC"
            with m.State("EXEC"):
                m.d.sync += [Cat(self.cnt, self.cnt_ovf).eq(self.cnt_next)]
                with m.If(self.cnt_next[:-1] == self.cnt_target):
                    m.next = "INIT"
                    m.d.sync += [
                        self.is_exec.eq(0),
                    ]

        return m


if __name__ == "__main__":
    num_bits = 4
    acc_bits = 8
    cnt_bits = 3
    signed = True
    dut = PE(
        num_bits=num_bits, acc_bits=acc_bits, cnt_bits=cnt_bits, signed=signed
    )
    dut = ResetInserter(dut.in_rst)(dut)

    from amaranth.sim import Simulator
    import numpy as np

    np.random.seed(8)

    def test_case(dut, init, a, b):
        yield dut.in_init.eq(init)
        yield dut.in_a.eq(int(a))
        yield dut.in_b.eq(int(b))
        yield

    def bench():
        test_cycles = 16
        for _ in range(test_cycles):
            init = np.random.randint(low=1, high=2**cnt_bits)
            rdm_init_cycle = np.random.randint(low=-1, high=4)
            for _ in range(init + rdm_init_cycle):
                rdm = np.random.randint(low=0, high=2**num_bits, size=[2])
                yield from test_case(dut, init, *rdm)
                init = 0

    from pathlib import Path

    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_clock(1e-6)
    sim.add_sync_process(bench)

    with open(p.with_suffix(".vcd"), "w") as f:
        with sim.write_vcd(f):
            sim.run()

    from amaranth.back import verilog

    top = PE(
        num_bits=num_bits, acc_bits=acc_bits, cnt_bits=cnt_bits, signed=signed
    )
    with open(p.with_suffix(".v"), "w") as f:
        f.write(
            verilog.convert(
                top,
                ports=[
                    top.in_a,
                    top.in_b,
                    top.in_init,
                    top.out_d,
                    top.out_d_valid,
                    top.out_ovf,
                ],
            )
        )
