def parse_structure(astr, level):
    if level == 0 :
        loopbeg = "/**begin repeat"
        loopend = "/**end repeat**/"
    else :
        loopbeg = "/**begin repeat%d" % level
        loopend = "/**end repeat%d**/" % level
    ind = 0
    line = 0
    spanlist = []
    while True:
        start = astr.find(loopbeg, ind)
        if start == -1:
            break
        start2 = astr.find("*/", start)
        start2 = astr.find("\n", start2)
        fini1 = astr.find(loopend, start2)
        fini2 = astr.find("\n", fini1)
        line += astr.count("\n", ind, start2+1)
        spanlist.append((start, start2+1, fini1, fini2+1, line))
        line += astr.count("\n", start2+1, fini2)
        ind = fini2
    spanlist.sort()
    return spanlist
def paren_repl(obj):
    torep = obj.group(1)
    numrep = obj.group(2)
    return ','.join([torep]*int(numrep))
parenrep = re.compile(r"[(]([^)]*)[)]\*(\d+)")
plainrep = re.compile(r"([^*]+)\*(\d+)")
def parse_values(astr):
    astr = parenrep.sub(paren_repl, astr)
    astr = ','.join([plainrep.sub(paren_repl, x.strip())
                     for x in astr.split(',')])
    return astr.split(',')
stripast = re.compile(r"\n\s*\*?")
exclude_vars_re = re.compile(r"(\w*)=(\w*)")
exclude_re = re.compile(":exclude:")
def parse_loop_header(loophead) :
    loophead = stripast.sub("", loophead)
    names = []
    reps = named_re.findall(loophead)
    nsub = None
    for rep in reps:
        name = rep[0]
        vals = parse_values(rep[1])
        size = len(vals)
        if nsub is None :
            nsub = size
        elif nsub != size :
            msg = "Mismatch in number of values, %d != %d\n%s = %s"
            raise ValueError(msg % (nsub, size, name, vals))
        names.append((name, vals))
    excludes = []
    for obj in exclude_re.finditer(loophead):
        span = obj.span()
        endline = loophead.find('\n', span[1])
        substr = loophead[span[1]:endline]
        ex_names = exclude_vars_re.findall(substr)
        excludes.append(dict(ex_names))
    dlist = []
    if nsub is None :
        raise ValueError("No substitution variables found")
    for i in range(nsub):
        tmp = {name: vals[i] for name, vals in names}
        dlist.append(tmp)
    return dlist
replace_re = re.compile(r"@([\w]+)@")
def parse_string(astr, env, level, line) :
    def replace(match):
        name = match.group(1)
        try :
            val = env[name]
        except KeyError:
            msg = 'line %d: no definition of key "%s"'%(line, name)
            raise ValueError(msg)
        return val
    code = [lineno]
    struct = parse_structure(astr, level)
    if struct :
        oldend = 0
        newlevel = level + 1
        for sub in struct:
            pref = astr[oldend:sub[0]]
            head = astr[sub[0]:sub[1]]
            text = astr[sub[1]:sub[2]]
            oldend = sub[3]
            newline = line + sub[4]
            code.append(replace_re.sub(replace, pref))
            try :
                envlist = parse_loop_header(head)
            except ValueError as e:
                msg = "line %d: %s" % (newline, e)
                raise ValueError(msg)
            for newenv in envlist :
                newenv.update(env)
                newcode = parse_string(text, newenv, newlevel, newline)
                code.extend(newcode)
        suff = astr[oldend:]
        code.append(replace_re.sub(replace, suff))
    else :
        code.append(replace_re.sub(replace, astr))
    code.append('\n')
    return ''.join(code)
def process_str(astr):
    code = [header]
    code.extend(parse_string(astr, global_names, 0, 1))
    return ''.join(code)
                            r"(?P<name>[\w\d./\\]+[.]src)['\"]", re.I)
def resolve_includes(source):
    d = os.path.dirname(source)
    with open(source) as fid:
        lines = []
        for line in fid:
            m = include_src_re.match(line)
            if m:
                fn = m.group('name')
                if not os.path.isabs(fn):
                    fn = os.path.join(d, fn)
                if os.path.isfile(fn):
                    print('Including file', fn)
                    lines.extend(resolve_includes(fn))
                else:
                    lines.append(line)
            else:
                lines.append(line)
    return lines
def process_file(source):
    lines = resolve_includes(source)
    sourcefile = os.path.normcase(source).replace("\\", "\\\\")
    try:
        code = process_str(''.join(lines))
    except ValueError as e:
        raise ValueError('In "%s" loop at %s' % (sourcefile, e))
def unique_key(adict):
    allkeys = list(adict.keys())
    done = False
    n = 1
    while not done:
        newkey = "".join([x[:n] for x in allkeys])
        if newkey in allkeys:
            n += 1
        else:
            done = True
    return newkey
def main():
    try:
        file = sys.argv[1]
    except IndexError:
        fid = sys.stdin
        outfile = sys.stdout
    else:
        fid = open(file, 'r')
        (base, ext) = os.path.splitext(file)
        newname = base
        outfile = open(newname, 'w')
    allstr = fid.read()
    try:
        writestr = process_str(allstr)
    except ValueError as e:
        raise ValueError("In %s loop at %s" % (file, e))
    outfile.write(writestr)
if __name__ == "__main__":
    main()
