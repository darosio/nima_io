
import subprocess
import javabridge
import os
import imgread.read as ir

datafolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


#     use capsys and capfd
# https://docs.pytest.org/en/2.8.7/capture.html
class Test_imgdiff:

    def setup_class(cls):
        ir.ensure_VM()
        cls.fp_a = os.path.join(datafolder, 'im1s1z3c5t_a.ome.tif')
        cls.fp_b = os.path.join(datafolder, 'im1s1z3c5t_b.ome.tif')
        cls.fp_bmd = os.path.join(datafolder, 'im1s1z2c5t_bmd.ome.tif')
        cls.fp_bpix = os.path.join(datafolder, 'im1s1z3c5t_bpix.ome.tif')

    def teardown_class(cls):
        print("Killing VirtualMachine")
        javabridge.kill_vm()

    # FIXED it was checking same thing twise.
    # def test_diff(self):
    #     assert ir.diff(self.fp_a, self.fp_b)
    #     assert not ir.diff(self.fp_a, self.fp_bmd)
    #     assert not ir.diff(self.fp_a, self.fp_bpix)

    def test_script(self):
        cmd_line = ['imgdiff', self.fp_a, self.fp_b]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
        assert p.communicate()[0] == b"Files seem equal.\n"
        cmd_line = ['imgdiff', self.fp_a, self.fp_bmd]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
        assert p.communicate()[0] == b"Files differ.\n"
        cmd_line = ['imgdiff', self.fp_a, self.fp_bpix]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
        assert p.communicate()[0] == b"Files differ.\n"
