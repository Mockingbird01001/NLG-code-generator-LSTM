
import bisect
def intranges_from_list(list_):
    sorted_list = sorted(list_)
    ranges = []
    last_write = -1
    for i in range(len(sorted_list)):
        if i+1 < len(sorted_list):
            if sorted_list[i] == sorted_list[i+1]-1:
                continue
        current_range = sorted_list[last_write+1:i+1]
        ranges.append(_encode_range(current_range[0], current_range[-1] + 1))
        last_write = i
    return tuple(ranges)
def _encode_range(start, end):
    return (start << 32) | end
def _decode_range(r):
    return (r >> 32), (r & ((1 << 32) - 1))
def intranges_contain(int_, ranges):
    tuple_ = _encode_range(int_, 0)
    pos = bisect.bisect_left(ranges, tuple_)
    if pos > 0:
        left, right = _decode_range(ranges[pos-1])
        if left <= int_ < right:
            return True
    if pos < len(ranges):
        left, _ = _decode_range(ranges[pos])
        if left == int_:
            return True
    return False
