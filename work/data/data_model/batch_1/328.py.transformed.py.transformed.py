
import itertools
import operator
from numpy.core.multiarray import c_einsum
from numpy.core.numeric import asanyarray, tensordot
from numpy.core.overrides import array_function_dispatch
__all__ = ['einsum', 'einsum_path']
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)
def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1
    return overall_size * op_factor
def _compute_size_by_dict(indices, idx_dict):
    ret = 1
    for i in indices:
        ret *= idx_dict[i]
    return ret
def _find_contraction(positions, input_sets, output_set):
    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value
    new_result = idx_remain & idx_contract
    idx_removed = (idx_contract - new_result)
    remaining.append(new_result)
    return (new_result, remaining, idx_removed, idx_contract)
def _optimal_path(input_sets, output_set, idx_dict, memory_limit):
    full_results = [(0, [], input_sets)]
    for iteration in range(len(input_sets) - 1):
        iter_results = []
        for curr in full_results:
            cost, positions, remaining = curr
            for con in itertools.combinations(range(len(input_sets) - iteration), 2):
                cont = _find_contraction(con, remaining, output_set)
                new_result, new_input_sets, idx_removed, idx_contract = cont
                new_size = _compute_size_by_dict(new_result, idx_dict)
                if new_size > memory_limit:
                    continue
                total_cost =  cost + _flop_count(idx_contract, idx_removed, len(con), idx_dict)
                new_pos = positions + [con]
                iter_results.append((total_cost, new_pos, new_input_sets))
        if iter_results:
            full_results = iter_results
        else:
            path = min(full_results, key=lambda x: x[0])[1]
            path += [tuple(range(len(input_sets) - iteration))]
            return path
    if len(full_results) == 0:
        return [tuple(range(len(input_sets)))]
    path = min(full_results, key=lambda x: x[0])[1]
    return path
def _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost):
    contract = _find_contraction(positions, input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract
    new_size = _compute_size_by_dict(idx_result, idx_dict)
    if new_size > memory_limit:
        return None
    old_sizes = (_compute_size_by_dict(input_sets[p], idx_dict) for p in positions)
    removed_size = sum(old_sizes) - new_size
    cost = _flop_count(idx_contract, idx_removed, len(positions), idx_dict)
    sort = (-removed_size, cost)
    if (path_cost + cost) > naive_cost:
        return None
    return [sort, positions, new_input_sets]
def _update_other_results(results, best):
    best_con = best[1]
    bx, by = best_con
    mod_results = []
    for cost, (x, y), con_sets in results:
        if x in best_con or y in best_con:
            continue
        del con_sets[by - int(by > x) - int(by > y)]
        del con_sets[bx - int(bx > x) - int(bx > y)]
        con_sets.insert(-1, best[2][-1])
        mod_con = x - int(x > bx) - int(x > by), y - int(y > bx) - int(y > by)
        mod_results.append((cost, mod_con, con_sets))
    return mod_results
def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
    if len(input_sets) == 1:
        return [(0,)]
    elif len(input_sets) == 2:
        return [(0, 1)]
    contract = _find_contraction(range(len(input_sets)), input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract
    naive_cost = _flop_count(idx_contract, idx_removed, len(input_sets), idx_dict)
    comb_iter = itertools.combinations(range(len(input_sets)), 2)
    known_contractions = []
    path_cost = 0
    path = []
    for iteration in range(len(input_sets) - 1):
        for positions in comb_iter:
            if input_sets[positions[0]].isdisjoint(input_sets[positions[1]]):
                continue
            result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost,
                                                 naive_cost)
            if result is not None:
                known_contractions.append(result)
        if len(known_contractions) == 0:
            for positions in itertools.combinations(range(len(input_sets)), 2):
                result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit,
                                                     path_cost, naive_cost)
                if result is not None:
                    known_contractions.append(result)
            if len(known_contractions) == 0:
                path.append(tuple(range(len(input_sets))))
                break
        best = min(known_contractions, key=lambda x: x[0])
        known_contractions = _update_other_results(known_contractions, best)
        input_sets = best[2]
        new_tensor_pos = len(input_sets) - 1
        comb_iter = ((i, new_tensor_pos) for i in range(new_tensor_pos))
        path.append(best[1])
        path_cost += best[0][1]
    return path
def _can_dot(inputs, result, idx_removed):
    if len(idx_removed) == 0:
        return False
    if len(inputs) != 2:
        return False
    input_left, input_right = inputs
    for c in set(input_left + input_right):
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False
        if nl + nr - 1 == int(c in result):
            return False
    set_left = set(input_left)
    set_right = set(input_right)
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed
    rs = len(idx_removed)
    if input_left == input_right:
        return True
    if set_left == set_right:
        return False
    if input_left[-rs:] == input_right[:rs]:
        return True
    if input_left[:rs] == input_right[-rs:]:
        return True
    if input_left[-rs:] == input_right[-rs:]:
        return True
    if input_left[:rs] == input_right[:rs]:
        return True
    if not keep_left or not keep_right:
        return False
    return True
def _parse_einsum_input(operands):
    if len(operands) == 0:
        raise ValueError("No input operands")
    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = [asanyarray(v) for v in operands[1:]]
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))
        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [asanyarray(v) for v in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError("For this input type lists must contain "
                                        "either int or Ellipsis") from e
                    subscripts += einsum_symbols[s]
            if num != last:
                subscripts += ","
        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError("For this input type lists must contain "
                                        "either int or Ellipsis") from e
                    subscripts += einsum_symbols[s]
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0
        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(',')
            out_sub = False
        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= (len(sub) - 3)
                if ellipse_count > longest:
                    longest = ellipse_count
                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)
        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]
        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) -
                                         set(out_ellipse)))
            subscripts += "->" + out_ellipse + normal_inds
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s
    for char in output_subscript:
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input"
                             % char)
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the "
                         "number of operands.")
    return (input_subscripts, output_subscript, operands)
def _einsum_path_dispatcher(*operands, optimize=None, einsum_call=None):
    return operands
@array_function_dispatch(_einsum_path_dispatcher, module='numpy')
def einsum_path(*operands, optimize='greedy', einsum_call=False):
    path_type = optimize
    if path_type is True:
        path_type = 'greedy'
    if path_type is None:
        path_type = False
    memory_limit = None
    if (path_type is False) or isinstance(path_type, str):
        pass
    elif len(path_type) and (path_type[0] == 'einsum_path'):
        pass
    elif ((len(path_type) == 2) and isinstance(path_type[0], str) and
            isinstance(path_type[1], (int, float))):
        memory_limit = int(path_type[1])
        path_type = path_type[0]
    else:
        raise TypeError("Did not understand the path: %s" % str(path_type))
    einsum_call_arg = einsum_call
    input_subscripts, output_subscript, operands = _parse_einsum_input(operands)
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))
    dimension_dict = {}
    broadcast_indices = [[] for x in range(len(input_list))]
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError("Einstein sum subscript %s does not contain the "
                             "correct number of indices for operand %d."
                             % (input_subscripts[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if dim == 1:
                broadcast_indices[tnum].append(char)
            if char in dimension_dict.keys():
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError("Size of label '%s' for operand %d (%d) "
                                     "does not match previous terms (%d)."
                                     % (char, tnum, dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim
    broadcast_indices = [set(x) for x in broadcast_indices]
    size_list = [_compute_size_by_dict(term, dimension_dict)
                 for term in input_list + [output_subscript]]
    max_size = max(size_list)
    if memory_limit is None:
        memory_arg = max_size
    else:
        memory_arg = memory_limit
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = _flop_count(indices, inner_product, len(input_list), dimension_dict)
    if (path_type is False) or (len(input_list) in [1, 2]) or (indices == output_set):
        path = [tuple(range(len(input_list)))]
    elif path_type == "greedy":
        path = _greedy_path(input_sets, output_set, dimension_dict, memory_arg)
    elif path_type == "optimal":
        path = _optimal_path(input_sets, output_set, dimension_dict, memory_arg)
    elif path_type[0] == 'einsum_path':
        path = path_type[1:]
    else:
        raise KeyError("Path name %s not found", path_type)
    cost_list, scale_list, size_list, contraction_list = [], [], [], []
    for cnum, contract_inds in enumerate(path):
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))
        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract
        cost = _flop_count(idx_contract, idx_removed, len(contract_inds), dimension_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(_compute_size_by_dict(out_inds, dimension_dict))
        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))
            bcast |= broadcast_indices.pop(x)
        new_bcast_inds = bcast - idx_removed
        if not len(idx_removed & bcast):
            do_blas = _can_dot(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])
        input_list.append(idx_result)
        broadcast_indices.append(new_bcast_inds)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result
        contraction = (contract_inds, idx_removed, einsum_str, input_list[:], do_blas)
        contraction_list.append(contraction)
    opt_cost = sum(cost_list) + 1
    if einsum_call_arg:
        return (operands, contraction_list)
    overall_contraction = input_subscripts + "->" + output_subscript
    header = ("scaling", "current", "remaining")
    speedup = naive_cost / opt_cost
    max_i = max(size_list)
    path_print  = "  Complete contraction:  %s\n" % overall_contraction
    path_print += "         Naive scaling:  %d\n" % len(indices)
    path_print += "     Optimized scaling:  %d\n" % max(scale_list)
    path_print += "      Naive FLOP count:  %.3e\n" % naive_cost
    path_print += "  Optimized FLOP count:  %.3e\n" % opt_cost
    path_print += "   Theoretical speedup:  %3.3f\n" % speedup
    path_print += "  Largest intermediate:  %.3e elements\n" % max_i
    path_print += "-" * 74 + "\n"
    path_print += "%6s %24s %40s\n" % header
    path_print += "-" * 74
    for n, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        remaining_str = ",".join(remaining) + "->" + output_subscript
        path_run = (scale_list[n], einsum_str, remaining_str)
        path_print += "\n%4d    %24s %40s" % path_run
    path = ['einsum_path'] + path
    return (path, path_print)
def _einsum_dispatcher(*operands, out=None, optimize=None, **kwargs):
    yield from operands
    yield out
@array_function_dispatch(_einsum_dispatcher, module='numpy')
def einsum(*operands, out=None, optimize=False, **kwargs):
    specified_out = out is not None
    if optimize is False:
        if specified_out:
            kwargs['out'] = out
        return c_einsum(*operands, **kwargs)
    valid_einsum_kwargs = ['dtype', 'order', 'casting']
    unknown_kwargs = [k for (k, v) in kwargs.items() if
                      k not in valid_einsum_kwargs]
    if len(unknown_kwargs):
        raise TypeError("Did not understand the following kwargs: %s"
                        % unknown_kwargs)
    operands, contraction_list = einsum_path(*operands, optimize=optimize,
                                             einsum_call=True)
    output_order = kwargs.pop('order', 'K')
    if output_order.upper() == 'A':
        if all(arr.flags.f_contiguous for arr in operands):
            output_order = 'F'
        else:
            output_order = 'C'
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        tmp_operands = [operands.pop(x) for x in inds]
        handle_out = specified_out and ((num + 1) == len(contraction_list))
        if blas:
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')
            tensor_result = input_left + input_right
            for s in idx_rm:
                tensor_result = tensor_result.replace(s, "")
            left_pos, right_pos = [], []
            for s in sorted(idx_rm):
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))
            new_view = tensordot(*tmp_operands, axes=(tuple(left_pos), tuple(right_pos)))
            if (tensor_result != results_index) or handle_out:
                if handle_out:
                    kwargs["out"] = out
                new_view = c_einsum(tensor_result + '->' + results_index, new_view, **kwargs)
        else:
            if handle_out:
                kwargs["out"] = out
            new_view = c_einsum(einsum_str, *tmp_operands, **kwargs)
        operands.append(new_view)
        del tmp_operands, new_view
    if specified_out:
        return out
    else:
        return asanyarray(operands[0], order=output_order)
