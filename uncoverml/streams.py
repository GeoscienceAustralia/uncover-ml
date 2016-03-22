from threading import Thread
from queue import Queue, Empty


# Adapted from gist: https://gist.github.com/EyalAr/7915597
class NonBlockingStreamReader:

    def __init__(self, stream):
        """
        A Non-blocking stream reader class for capturing stdout/stderr streams.

        Parameters
        ----------
        stream: stream
            the stream to read from. Usually a process' stdout or stderr.
        """

        self._s = stream
        self._q = Queue()

        def _populateQueue(stream, queue):
            """
            Collect lines from 'stream' and put them in 'queque'.
            """

            while True:
                line = stream.readline()
                if line:
                    queue.put(line)
                else:
                    break  # End of stream
                    # raise UnexpectedEndOfStream

        self._t = Thread(target=_populateQueue, args=(self._s, self._q))
        self._t.daemon = True
        self._t.start()  # start collecting lines from the stream

    def readline(self, timeout=None):
        """
        Read a line from the stream, without blocking.

        Parameters
        ----------
        timeout: float, optional
            time in seconds to wait for stream to be populated before
            reading. None for read immediately.

        Returns
        -------
        bytes or str:
            depending on the underlying stream. If this times out, None is
            returned.
        """
        try:
            return self._q.get(block=timeout is not None, timeout=timeout)
        except Empty:
            return None


class UnexpectedEndOfStream(Exception):
        pass

