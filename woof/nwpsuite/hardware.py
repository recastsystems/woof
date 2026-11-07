"""Context-free box inventory and check receipt normalization."""
from pathlib import Path
import shutil
import socket
import subprocess
from .registry import SuiteRefusal, read_json

def inventory():
    from woof import __version__
    from woof.core.streaming import _host_total_bytes
    from tilestream.autoplan import _host_memavailable
    devices=[]
    error=None
    try:
        command=[*__import__('woof.gpu_smi', fromlist=['command']).command(),"--query-gpu=index,uuid,name,memory.total","--format=csv,noheader,nounits"]
        output=subprocess.run(command,text=True,capture_output=True,check=True,timeout=15).stdout
        for line in output.splitlines():
            index,uuid,name,memory=[part.strip() for part in line.split(",")]
            devices.append(dict(index=int(index),uuid=uuid,name=name,memory_total_bytes=int(memory)*1048576))
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        error=str(exc)
    return dict(schema="gpuwm.box.check.v1",host=socket.gethostname(),gpuwm_version=__version__,devices=devices,
                host_ram_bytes=_host_total_bytes(),host_available_bytes=_host_memavailable(),
                disk_free_bytes=shutil.disk_usage(Path.cwd()).free,device_query_error=error,
                readiness=dict(collected=False,ready=None),basis="nvidia-smi inventory only; no CUDA context")

def box_from_check(path, identifier):
    data=read_json(path)
    devices=data.get("devices",[])
    if not isinstance(devices,list) or not devices:
        raise SuiteRefusal(f"{path}: no device inventory in check JSON; prevents guessing card capacity")
    cards=[]
    for device in devices:
        memory=device.get("memory_total_bytes")
        if not memory or not device.get("name"):
            raise SuiteRefusal(f"{path}: incomplete device inventory {device.get('index')}")
        cards.append(dict(index=device["index"],uuid=device.get("uuid",""),name=device["name"],memory_mib=memory/1048576))
    return dict(id=identifier,host=data.get("host","local"),check=str(Path(path).resolve()),cards=len(cards),
                card_model=cards[0]["name"],memory_mib=min(c["memory_mib"] for c in cards),inventory=cards,
                host_memory_mib=(data.get("host_ram_bytes") or 0)/1048576,disk_free_gb=(data.get("disk_free_bytes") or 0)/1e9,
                build=data.get("gpuwm_version","unknown"))
