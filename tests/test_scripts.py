"""
Module to test command-line scripts.
"""
import os
import subprocess

datafolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# pytestmark = pytest.mark.usefixtures("jvm")


# @pytest.fixture
# def jvm():
#     """Ensure a running JVM."""
#     ir.ensure_VM()
#     yield
#     javabridge.kill_vm()


# def setup_module():
#     """Ensure a running JVM."""
#     ir.ensure_VM()


# def teardown_module():
#     """Try to detach from the JVM as we cannot kill it.
#     https://github.com/LeeKamentsky/python-javabridge/issues/88
#     """
#     print("Killing VirtualMachine")
#     javabridge.kill_vm()


class Test_imgdiff:
    """
    Class testing imgdiff command using os.system/subprocess invocations and
    so without calling specific methods/units within nima_io package.
    """

    @classmethod
    def setup_class(cls):
        """Define data files for testing imgdiff."""
        cls.fp_a = os.path.join(datafolder, "im1s1z3c5t_a.ome.tif")
        cls.fp_b = os.path.join(datafolder, "im1s1z3c5t_b.ome.tif")
        cls.fp_bmd = os.path.join(datafolder, "im1s1z2c5t_bmd.ome.tif")
        cls.fp_bpix = os.path.join(datafolder, "im1s1z3c5t_bpix.ome.tif")

    def test_equal_files(self, capfd):
        "Test equal files."
        os.system("imgdiff {} {}".format(self.fp_a, self.fp_b))
        out, _ = capfd.readouterr()
        assert out == "Files seem equal.\n"

    def test_different_files(self):
        "Test different files."
        cmd_line = ["imgdiff", self.fp_a, self.fp_bmd]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
        assert p.communicate()[0] == b"Files differ.\n"

    def test_singlepixeldifferent_files(self):
        "Test different pixels data, same metadata."
        cmd_line = ["imgdiff", self.fp_a, self.fp_bpix]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
        assert p.communicate()[0] == b"Files differ.\n"
