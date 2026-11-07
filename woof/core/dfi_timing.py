"""GPU events embedded in a finite DFI graph; queried only after its pass."""
import ctypes as C
import os


class PassTiming:
    def __init__(self, stream, count):
        import cupy as cp
        self.stream = stream
        self.events = [tuple(cp.cuda.Event() for _ in range(3)) for _ in range(count)]
        self.tail = (cp.cuda.Event(),cp.cuda.Event())
        self.driver = C.WinDLL('nvcuda.dll') if os.name == 'nt' else C.CDLL('libcuda.so.1')
        self.driver.cuEventRecordWithFlags.argtypes = (C.c_void_p,C.c_void_p,C.c_uint)
        self.driver.cuEventRecordWithFlags.restype = C.c_int

    def record(self, event):
        code = self.driver.cuEventRecordWithFlags(event.ptr,self.stream.ptr,1)
        if code:
            raise RuntimeError(f'DFI graph timing event failed ({code})')

    def result(self):
        import cupy as cp
        filters = [cp.cuda.get_elapsed_time(a,b) for a,b,_ in self.events]
        solves = [cp.cuda.get_elapsed_time(b,c) for _,b,c in self.events]
        return dict(filter_ms=filters,model_solve_ms=solves,
            endpoint_reset_ms=cp.cuda.get_elapsed_time(*self.tail),
            relative_to_same_model_solves=(sum(filters)+sum(solves))/sum(solves))


class TreePassTiming:
    """Independent solve/filter clocks for each domain and recursive endpoint."""
    def __init__(self, stream, nodes):
        import cupy as cp
        self.domains = {}
        self.endpoints = {}
        self.cursor = {}
        for node in nodes:
            gid = node.cfg.grid_id
            self.domains[gid] = PassTiming(stream,node.clock.interval.steps)
            calls = 1 if node.parent is None else node.parent.clock.interval.steps
            self.endpoints[gid] = [tuple(cp.cuda.Event() for _ in range(2))
                                   for _ in range(calls)]
            self.cursor[gid] = 0

    def step(self, node, point):
        timing = self.domains[node.cfg.grid_id]
        timing.record(timing.events[node.clock.step_count][point])

    def endpoint(self, node, point):
        gid = node.cfg.grid_id
        self.domains[gid].record(self.endpoints[gid][self.cursor[gid]][point])
        if point:
            self.cursor[gid] += 1

    def reset(self, node, point):
        timing = self.domains[node.cfg.grid_id]
        timing.record(timing.tail[point])

    def result(self):
        import cupy as cp
        result = {}
        for gid,timing in self.domains.items():
            value = timing.result()
            value['recursive_endpoint_filter_ms'] = [
                cp.cuda.get_elapsed_time(a,b) for a,b in self.endpoints[gid]]
            result[str(gid)] = value
        return result
