import threading
import time

import pytest

from dataeval.utils._internal import read_ordered

pytestmark = pytest.mark.required


class TestSerialPath:
    @pytest.mark.parametrize("threads", [None, 1])
    def test_it_is_the_plain_loop_on_the_calling_thread(self, threads):
        seen: list[tuple[int, int]] = []

        def read(index: int) -> int:
            seen.append((index, threading.get_ident()))
            return index * 2

        assert list(read_ordered(read, range(5), threads)) == [0, 2, 4, 6, 8]
        assert seen == [(i, threading.get_ident()) for i in range(5)]

    def test_one_index_reads_on_the_calling_thread_whatever_the_thread_count(self):
        seen: list[int] = []
        assert list(read_ordered(lambda i: seen.append(threading.get_ident()) or i, [7], 8)) == [7]
        assert seen == [threading.get_ident()]

    @pytest.mark.parametrize("threads", [0, -1])
    def test_a_thread_count_below_one_raises_before_anything_is_read(self, threads):
        reads: list[int] = []
        with pytest.raises(ValueError, match="threads"):
            read_ordered(reads.append, range(3), threads)
        assert reads == []


class TestThreadedPath:
    def test_results_come_back_in_index_order_whatever_order_the_reads_finish(self):
        def read(index: int) -> int:
            time.sleep(0.005 * (7 - index))  # the last index finishes first
            return index

        assert list(read_ordered(read, range(8), 4)) == list(range(8))

    def test_reads_run_on_more_than_one_thread(self):
        idents: set[int] = set()

        def read(index: int) -> int:
            idents.add(threading.get_ident())
            time.sleep(0.01)
            return index

        list(read_ordered(read, range(8), 4))
        assert len(idents) > 1

    def test_the_read_ahead_never_runs_further_than_asked(self):
        started: list[int] = []
        lock = threading.Lock()

        def read(index: int) -> int:
            with lock:
                started.append(index)
            return index

        for consumed, _ in enumerate(read_ordered(read, range(40), 4, ahead=3)):
            time.sleep(0.002)  # let the workers get as far ahead as they may
            with lock:
                assert max(started) <= consumed + 3

    def test_an_exception_is_raised_at_that_items_position_and_later_reads_are_not_started(self):
        started: list[int] = []

        def read(index: int) -> int:
            started.append(index)
            if index == 3:
                raise RuntimeError("bad item")
            time.sleep(0.01)
            return index

        collected: list[int] = []
        with pytest.raises(RuntimeError, match="bad item"):
            collected.extend(read_ordered(read, range(100), 2, ahead=4))
        assert collected == [0, 1, 2]
        assert max(started) < 20

    def test_no_thread_is_left_running_afterward(self):
        before = threading.active_count()
        list(read_ordered(lambda i: i, range(16), 4))
        with pytest.raises(RuntimeError):
            list(read_ordered(lambda i: (_ for _ in ()).throw(RuntimeError("x")) if i == 2 else i, range(16), 4))
        assert threading.active_count() == before

    def test_closing_the_generator_early_stops_reading(self):
        started: list[int] = []

        def read(index: int) -> int:
            started.append(index)
            time.sleep(0.005)
            return index

        generator = read_ordered(read, range(1000), 2, ahead=4)
        next(generator)
        generator.close()
        time.sleep(0.05)
        assert max(started) < 20
