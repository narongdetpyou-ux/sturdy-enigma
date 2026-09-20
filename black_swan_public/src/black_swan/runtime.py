"""Bounded local runtime: per-worker pipes, active deadlines, kill/restart.

Per-worker transport avoids a killed worker corrupting a shared queue lock.
This is a Python prototype, NOT an HTTP service or an untrusted-pickle boundary.
Only pass primitive ScenarioInput/WindowUpdate data. close() stops all workers.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import heapq
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import threading
import time

from .engine import BlackSwanV8, ScenarioInput, WindowUpdate, ContextEvent, fallback


def transport_preflight(item):
    """Bound shapes/types BEFORE serialization; do not pickle arbitrary objects.

    Numeric finiteness, calibrated limits, history quality and trust are still
    checked in the isolated engine. This duplicate boundary has measured cost.
    """
    def text_ok(x, empty=False):
        return type(x) is str and len(x) <= 256 and (empty or bool(x))
    if type(item) not in (ScenarioInput, WindowUpdate) or not text_ok(item.scenario_id):
        return False
    if type(item) is WindowUpdate and not text_ok(item.baseline_id):
        return False
    series = [(item.recent, 256)]
    statuses = [item.recent_status]
    if type(item) is ScenarioInput:
        series.append((item.history,4096))
        statuses.append(item.history_status)
    total = 0
    for mapping, limit in series:
        if type(mapping) is not dict or len(mapping) > 256:
            return False
        for key, values in mapping.items():
            if not text_ok(key) or type(values) is not list or len(values) > limit:
                return False
            total += len(values)
            if total > 262144:
                return False
            for value in values:
                if value is None or type(value) is float:
                    continue
                if type(value) is not int or value.bit_length() > 340:
                    return False
    for mapping in statuses:
        if type(mapping) is not dict or len(mapping)>256:
            return False
        for key, values in mapping.items():
            if not text_ok(key) or type(values) is not list or len(values)>4096:
                return False
            if not all(text_ok(x) for x in values):
                return False
    if type(item.lifecycle) is not dict or len(item.lifecycle)>256:
        return False
    if not all(text_ok(k) and text_ok(v) for k,v in item.lifecycle.items()):
        return False
    if type(item.context_events) is not list or len(item.context_events)>32:
        return False
    for event in item.context_events:
        if (type(event) is not ContextEvent or not text_ok(event.event_key)
                or not text_ok(event.evidence_ref,empty=True)
                or type(event.covered_metrics) is not tuple or len(event.covered_metrics)>256
                or not all(text_ok(x) for x in event.covered_metrics)
                or type(event.approved) is not bool or type(event.active) is not bool
                or type(event.confidence) not in (int,float)):
            return False
        if type(event.confidence) is int and event.confidence.bit_length()>340:
            return False
    return True


def _worker(connection, calibration, trusted, baselines, test_faults):
    model = BlackSwanV8(calibration, trusted_contexts=trusted)
    for name, item in baselines:
        model.register_baseline(name, item)
    connection.send(('ready', os.getpid()))
    try:
        while True:
            job_id, item, deadline, fault = connection.recv()
            started = time.monotonic()
            if started >= deadline:
                decision = fallback('TIMEOUT_REVIEW', 'QUEUE_DEADLINE_EXCEEDED')
            else:
                try:
                    if test_faults and fault == 'stall':
                        time.sleep(60)
                    elif test_faults and fault == 'crash':
                        os._exit(77)
                    elif test_faults and fault == 'exception':
                        raise RuntimeError('inert fault injection')
                    decision = model.decide_update(item) if type(item) is WindowUpdate else model.decide(item)
                except Exception as exc:
                    decision = fallback('WORKER_ERROR_REVIEW', f'WORKER_EXCEPTION_{type(exc).__name__}')
            connection.send(('done', job_id, started, time.monotonic(), decision))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


@dataclass
class Ticket:
    job_id: int
    accepted: bool
    submitted: float
    future: Future


class IsolatedDecisionPool:
    def __init__(self, calibration, *, workers=4, queue_capacity=128,
                 deadline_s=.5, trusted_contexts=(), baselines=(), test_faults=False):
        if not 1 <= workers <= 32 or not 1 <= queue_capacity <= 10000 or not .01 <= deadline_s <= 60:
            raise ValueError('invalid runtime limits')
        self.ctx = mp.get_context('spawn')
        self.worker_count, self.capacity, self.deadline_s = workers, queue_capacity, deadline_s
        self.config = (calibration, tuple(trusted_contexts), tuple(baselines), test_faults)
        self.test_faults = test_faults
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._pending, self._workers = {}, {}
        self._waiting, self._deadlines, self._next_id = deque(), [], 0
        self._stats = {'offered': 0, 'accepted': 0, 'rejected': 0, 'completed': 0,
                       'timeouts': 0, 'cancelled': 0, 'worker_restarts': 0, 'peak_pending': 0, 'peak_queue': 0}
        self.monitor_error = None
        try:
            for index in range(workers):
                self._spawn(index)
            for row in self._workers.values():
                if not row['connection'].poll(20) or row['connection'].recv()[0] != 'ready':
                    raise RuntimeError('worker failed startup')
                row['ready'] = True
        except Exception:
            for index in list(self._workers):
                self._terminate(index)
            raise
        self._monitor = threading.Thread(target=self._watch, daemon=True, name='black-swan-watchdog')
        self._monitor.start()

    def _spawn(self, index):
        parent, child = self.ctx.Pipe(duplex=True)
        process = self.ctx.Process(target=_worker, args=(child, *self.config), daemon=True)
        process.start()
        child.close()
        self._workers[index] = {'process': process, 'connection': parent, 'ready': False, 'job': None}

    def _terminate(self, index):
        row = self._workers.pop(index)
        if row['process'].is_alive():
            row['process'].kill()
        row['process'].join(timeout=.2)
        row['connection'].close()

    def _restart(self, index):
        self._terminate(index)
        self._stats['worker_restarts'] += 1
        if not self._stop.is_set():
            self._spawn(index)

    def submit(self, item, *, test_fault=None):
        if test_fault is not None and (not self.test_faults or test_fault not in ('stall', 'crash', 'exception')):
            raise ValueError('fault injection disabled or invalid')
        now, future = time.monotonic(), Future()
        safe_transport = transport_preflight(item)
        with self._lock:
            self._next_id += 1
            job_id = self._next_id
            self._stats['offered'] += 1
            rejection = None
            if self._stop.is_set():
                rejection = fallback('CANCELLED_REVIEW', 'RUNTIME_CLOSED')
            elif not safe_transport:
                rejection = fallback('INVALID_INPUT_REVIEW', 'UNSAFE_OR_OVERSIZED_TRANSPORT_INPUT')
            elif len(self._pending) >= self.capacity + self.worker_count:
                rejection = fallback('OVERLOAD_REVIEW', 'PENDING_CAPACITY_EXCEEDED')
            elif len(self._waiting) >= self.capacity:
                rejection = fallback('OVERLOAD_REVIEW', 'QUEUE_FULL')
            if rejection:
                self._stats['rejected'] += 1
                future.set_result(self._result(rejection, now, now, now, False))
                return Ticket(job_id, False, now, future)
            deadline = now + self.deadline_s
            self._pending[job_id] = {'future': future, 'submitted': now, 'deadline': deadline,
                                     'started': None, 'worker': None}
            self._waiting.append((job_id, item, deadline, test_fault))
            heapq.heappush(self._deadlines, (deadline, job_id))
            self._stats['accepted'] += 1
            self._stats['peak_pending'] = max(self._stats['peak_pending'], len(self._pending))
            self._stats['peak_queue'] = max(self._stats['peak_queue'], len(self._waiting))
            return Ticket(job_id, True, now, future)

    @staticmethod
    def _result(decision, submitted, started, finished, accepted):
        now = time.monotonic()
        return {'decision': decision, 'accepted': accepted,
                'submitted': submitted, 'started': started, 'finished': finished, 'completed': now,
                'queue_ms': max(0, (started - submitted) * 1000),
                'compute_ms': max(0, (finished - started) * 1000), 'end_to_end_ms': (now - submitted) * 1000}

    def _complete(self, job_id, decision, started=None, finished=None):
        record = self._pending.pop(job_id, None)
        if record is None:
            return
        now = time.monotonic()
        started = started if started is not None else (record['started'] or now)
        finished = finished if finished is not None else now
        if not record['future'].done():
            record['future'].set_result(self._result(decision, record['submitted'], started, finished, True))
        self._stats['completed'] += 1
        if decision.final_route == 'TIMEOUT_REVIEW':
            self._stats['timeouts'] += 1
        elif decision.final_route == 'CANCELLED_REVIEW':
            self._stats['cancelled'] += 1

    def cancel(self, job_id):
        """Cancel queued/running work, emit terminal REVIEW, and kill active compute."""
        with self._lock:
            record = self._pending.get(job_id)
            if record is None:
                return False
            index = record['worker']
            self._complete(job_id, fallback('CANCELLED_REVIEW', 'CALLER_CANCELLED'))
            if index is not None and self._workers[index]['job']==job_id:
                self._restart(index)
            return True

    def _watch(self):
        try:
            while not self._stop.is_set():
                with self._lock:
                    connections = {row['connection']: index for index, row in self._workers.items()}
                ready = wait(list(connections), timeout=.001)
                with self._lock:
                    for connection in ready:
                        index = connections[connection]
                        row = self._workers[index]
                        try:
                            message = connection.recv()
                            if message[0] == 'ready':
                                row['ready'] = True
                                continue
                            _, job_id, started, finished, decision = message
                            record = self._pending.get(job_id)
                            if record and time.monotonic() > record['deadline']:
                                decision = fallback('TIMEOUT_REVIEW', 'RESPONSE_DEADLINE_EXCEEDED')
                            self._complete(job_id, decision, started, finished)
                            row['job'] = None
                        except (EOFError, BrokenPipeError, OSError):
                            if row['job'] is not None:
                                self._complete(row['job'], fallback('WORKER_ERROR_REVIEW', 'WORKER_PROCESS_EXITED'))
                            self._restart(index)
                    now = time.monotonic()
                    while self._deadlines and self._deadlines[0][0] <= now:
                        _, job_id = heapq.heappop(self._deadlines)
                        record = self._pending.get(job_id)
                        if record is None:
                            continue
                        index = record['worker']
                        self._complete(job_id, fallback('TIMEOUT_REVIEW', 'HARD_DEADLINE_EXCEEDED'))
                        if index is not None and self._workers[index]['job'] == job_id:
                            self._restart(index)
                    while self._waiting and self._waiting[0][0] not in self._pending:
                        self._waiting.popleft()
                    for index, row in list(self._workers.items()):
                        if not row['ready'] or row['job'] is not None or not self._waiting:
                            continue
                        job = self._waiting.popleft()
                        record = self._pending.get(job[0])
                        if record is None:
                            continue
                        record['worker'], record['started'] = index, time.monotonic()
                        row['job'] = job[0]
                        try:
                            row['connection'].send(job)
                        except (EOFError, BrokenPipeError, OSError):
                            self._complete(job[0], fallback('WORKER_ERROR_REVIEW', 'WORKER_TRANSPORT_FAILED'))
                            self._restart(index)
        except Exception as exc:
            self.monitor_error = type(exc).__name__
            self._stop.set()
            with self._lock:
                for job_id in list(self._pending):
                    self._complete(job_id, fallback('WORKER_ERROR_REVIEW', 'WATCHDOG_FAILED'))
                for index in list(self._workers):
                    self._terminate(index)

    def stats(self):
        with self._lock:
            return {**self._stats, 'pending': len(self._pending), 'queued': len(self._waiting),
                    'alive_workers': sum(row['process'].is_alive() for row in self._workers.values()),
                    'monitor_error': self.monitor_error}

    def close(self):
        self._stop.set()
        self._monitor.join(timeout=3)
        with self._lock:
            for job_id in list(self._pending):
                self._complete(job_id, fallback('CANCELLED_REVIEW', 'RUNTIME_CLOSED'))
            for index in list(self._workers):
                self._terminate(index)
            self._waiting.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
