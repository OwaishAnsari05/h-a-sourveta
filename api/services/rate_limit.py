import threading
import time
from collections import defaultdict,deque
from api.config import RATE_LIMIT_PER_MINUTE

_lock=threading.Lock()
_requests=defaultdict(deque)
_WINDOW=60.0

def allow_request(client_id:str,limit:int=RATE_LIMIT_PER_MINUTE)->bool:
    now=time.monotonic()
    key=client_id or "unknown"
    with _lock:
        q=_requests[key]
        cutoff=now-_WINDOW
        while q and q[0]<=cutoff: q.popleft()
        if len(q)>=max(1,limit): return False
        q.append(now)
        return True