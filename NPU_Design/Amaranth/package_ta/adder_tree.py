from amaranth import *
from utils import is_power_of_two


class AdderTree(Elaboratable):
    def __init__(self, acc_bits, fan_in, signed=True):
        self.acc_bits = acc_bits
        self.fan_in = fan_in
        self.signed = signed
        assert is_power_of_two(fan_in)
        assert fan_in >= 2

        self.in_data = Array(
            [
                Signal(
                    Shape(acc_bits, signed=signed), name=f"in_data_{fan_in}_{i}"
                )
                for i in range(fan_in)
            ]
        )
        self.in_ovf = Array(
            [Signal(1, name=f"in_ovf_{fan_in}_{i}") for i in range(fan_in)]
        )
        self.in_valid = Array(
            [Signal(1, name=f"in_valid_{fan_in}_{i}") for i in range(fan_in)]
        )
        self.out_d = Signal(Shape(acc_bits, signed=signed))
        self.out_ovf = Signal(1)
        self.out_valid = Signal(1)

        self.tmp_ovf = Signal(1)

        self.tree_l = None
        self.tree_r = None
        if fan_in > 2:
            self.tree_l = AdderTree(acc_bits, fan_in // 2, signed=signed)
            self.tree_r = AdderTree(acc_bits, fan_in // 2, signed=signed)

    def elaborate(self, platform):
        m = Module()

        if self.fan_in > 2:
            m.submodules.tree_l = tree_l = self.tree_l
            m.submodules.tree_r = tree_r = self.tree_r

            for i in range(self.fan_in):
                half = self.fan_in // 2
                if i < half:
                    m.d.comb += [
                        tree_l.in_data[i].eq(self.in_data[i]),
                        tree_l.in_ovf[i].eq(self.in_ovf[i]),
                        tree_l.in_valid[i].eq(self.in_valid[i]),
                    ]
                else:
                    m.d.comb += [
                        tree_r.in_data[i - half].eq(self.in_data[i]),
                        tree_r.in_ovf[i - half].eq(self.in_ovf[i]),
                        tree_r.in_valid[i - half].eq(self.in_valid[i]),
                    ]
            m.d.comb += [
                Cat(self.out_d, self.tmp_ovf).eq(tree_l.out_d + tree_r.out_d),
                self.out_valid.eq(tree_l.out_valid & tree_r.out_valid),
            ]
            if self.signed:
                m.d.comb += [
                    self.out_ovf.eq(
                        tree_l.out_ovf
                        | tree_r.out_ovf
                        | (
                            ~self.tree_l.out_d[-1]
                            & ~self.tree_r.out_d[-1]
                            & self.out_d[-1]
                        )
                        | (
                            self.tree_l.out_d[-1]
                            & self.tree_r.out_d[-1]
                            & ~self.out_d[-1]
                        )
                    )
                ]
            else:
                m.d.comb += [
                    self.out_ovf.eq(
                        self.tmp_ovf | tree_l.out_ovf | tree_r.out_ovf
                    ),
                ]
        else:
            m.d.comb += [
                Cat(self.out_d, self.tmp_ovf).eq(
                    self.in_data[0] + self.in_data[1]
                ),
                self.out_valid.eq(self.in_valid[0] & self.in_valid[1]),
            ]
            if self.signed:
                m.d.comb += [
                    self.out_ovf.eq(
                        self.in_ovf[0]
                        | self.in_ovf[1]
                        | (
                            ~self.in_data[0][-1]
                            & ~self.in_data[1][-1]
                            & self.out_d[-1]
                        )
                        | (
                            self.in_data[0][-1]
                            & self.in_data[1][-1]
                            & ~self.out_d[-1]
                        )
                    )
                ]
            else:
                m.d.comb += [
                    self.out_ovf.eq(
                        self.tmp_ovf | self.in_ovf[0] | self.in_ovf[1]
                    ),
                ]

        return m


if __name__ == "__main__":
    acc_bits = 8
    fan_in = 4
    signed = True
    dut = AdderTree(acc_bits=acc_bits, fan_in=fan_in, signed=signed)

    from amaranth.sim import Simulator, Settle, Delay
    import numpy as np

    np.random.seed(42)

    def test_case(dut, in_data, in_ovf, in_valid):
        for i, d in enumerate(in_data):
            yield dut.in_data[i].eq(int(d))
        for i, d in enumerate(in_ovf):
            yield dut.in_ovf[i].eq(bool(d))
        for i, d in enumerate(in_valid):
            yield dut.in_valid[i].eq(bool(d))
        yield Settle()
        yield Delay(1e-6)
        hw_sum = yield dut.out_d
        hw_ovf = yield dut.out_ovf
        if hw_ovf:
            assert (hw_sum - sum(in_data)) % (2**acc_bits) == 0
        else:
            if in_data[0] == 30 and in_data[1] == -59:
                print(hw_sum)
                print(sum(in_data))
                print(in_data)
                print(bin(hw_sum), bin(sum(in_data) & 0b11111111))

            assert hw_sum == sum(in_data)

    def bench():
        for _ in range(512):
            yield from test_case(
                dut,
                in_data=np.random.randint(
                    low=-(2 ** (acc_bits - 2)),
                    high=2 ** (acc_bits - 2),
                    size=fan_in,
                ),
                in_ovf=np.random.randint(low=0, high=1, size=fan_in),
                in_valid=np.random.randint(low=0, high=2, size=fan_in),
            )

    from pathlib import Path

    p = Path(__file__)

    sim = Simulator(dut)
    sim.add_process(bench)

    with open(p.with_suffix(".vcd"), "w") as f:
        with sim.write_vcd(f):
            sim.run()

    from amaranth.back import verilog

    top = AdderTree(acc_bits=acc_bits, fan_in=fan_in, signed=signed)
    with open(p.with_suffix(".v"), "w") as f:
        f.write(
            verilog.convert(
                top,
                ports=[
                    *top.in_data,
                    *top.in_ovf,
                    *top.in_valid,
                    top.out_d,
                    top.out_ovf,
                    top.out_valid,
                ],
            )
        )
