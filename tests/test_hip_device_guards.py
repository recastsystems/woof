"""Offline HIP images must expose the same entry points as runtime images."""
from pathlib import Path
import subprocess

from woof.hip_source import translate
from woof import hip_source


def test_offline_translation_retains_device_only_thompson_entry_points():
    source = (Path(__file__).parents[1] / 'woof/core/kernels/thompson.cu').read_text()
    translated = translate(source)
    # Preprocess the actual source without either compiler's RTC macro: this
    # reproduces HIPCC AOT dropping the level-parallel fallout entry points.
    header = Path(hip_source.__file__).with_name('hip_portable.h').read_text()
    device_source = translated.split(header + '\n', 1)[1]
    # Bytes on the pipe: a text-mode pipe turns "\n" into "\r\n" on Windows (T4).
    result = subprocess.run(['cpp', '-P', '-D__HIPCC__=1', '-'], input=device_source.encode('utf-8'),
                            capture_output=True, check=True)
    preprocessed = result.stdout.decode('utf-8', 'replace')
    assert 'void thompson_cloud_sediment_levels_64_with_density_and_masks(' in preprocessed
    assert 'void thompson_rain_sediment_levels_64_with_presence(' in preprocessed


def test_negative_rtc_guards_keep_the_actual_compiler_identity():
    source = '#if !defined(__CUDACC_RTC__)\n#include <stdint.h>\n#endif\n'
    translated = translate(source)
    assert '#if !defined(__HIPCC_RTC__)\n#include <stdint.h>' in translated
