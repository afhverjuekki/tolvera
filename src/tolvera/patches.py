"""Patches for third-party libraries.

Current patches:
- 'dill.source.findsource fails when in asyncio REPL' https://github.com/uqfoundation/dill/issues/627
"""

import linecache
import re
from inspect import (
    getblock,
    getfile,
    getmodule,
    getsourcefile,
    indentsize,
    isbuiltin,
    isclass,
    iscode,
    isframe,
    isfunction,
    ismethod,
    ismodule,
    istraceback,
)
from tokenize import TokenError

import dill
from dill._dill import IS_IPYTHON

# __all__ = ['findsource', 'getsourcelines', 'getsource', 'indent', 'outdent', \
#            '_wrap', 'dumpsource', 'getname', '_namespace', 'getimport', \
#            '_importable', 'importable','isdynamic', 'isfrommain']


def findsource(object):
    # print(f"[dill.source.findsource] PATCHED")

    """Return the entire source file and starting line number for an object.
    For interactively-defined objects, the 'file' is the interpreter's history.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a list of all the lines
    in the file and the line number indexes a line in that list.  An IOError
    is raised if the source code cannot be retrieved, while a TypeError is
    raised for objects where the source code is unavailable (e.g. builtins)."""

    def patched_getfile(module):
        # set file = None when module.__package__ == 'asyncio'
        # print(f"[dill.source.patched_getfile] module={module}\nmodule.__package__={module.__package__}\nmodule.__name__={module.__name__}")
        if module.__package__ == "asyncio":
            raise TypeError
        # if module.__package__ == 'sardine':
        #     raise TypeError
        ret = getfile(module)
        return ret

    module = getmodule(object)
    # try: file = getfile(module)
    try:
        file = patched_getfile(module)
    except TypeError:
        file = None
    # correctly compute `is_module_main` when in asyncio
    is_module_main = module and module.__name__ == "__main__" and not file
    # is_module_main = (module and module.__name__ == '__main__' or module.__name__ == 'sardine' and not file)
    print(
        f"[dill.source.findsource] module: {module}, file: {file}, is_module_main: {is_module_main}"
    )
    if IS_IPYTHON and is_module_main:
        # FIXME: quick fix for functions and classes in IPython interpreter
        try:
            file = getfile(object)
            sourcefile = getsourcefile(object)
        except TypeError:
            if isclass(object):
                for object_method in filter(isfunction, object.__dict__.values()):
                    # look for a method of the class
                    file_candidate = getfile(object_method)
                    if not file_candidate.startswith("<ipython-input-"):
                        continue
                    file = file_candidate
                    sourcefile = getsourcefile(object_method)
                    break
        if file:
            lines = linecache.getlines(file)
        else:
            # fallback to use history
            history = "\n".join(get_ipython().history_manager.input_hist_parsed)
            lines = [line + "\n" for line in history.splitlines()]
    # use readline when working in interpreter (i.e. __main__ and not file)
    elif is_module_main:
        try:
            import readline

            err = ""
        except ImportError:
            import sys

            err = sys.exc_info()[1].args[0]
            if sys.platform[:3] == "win":
                err += ", please install 'pyreadline'"
        if err:
            raise IOError(err)
        lbuf = readline.get_current_history_length()
        lines = [readline.get_history_item(i) + "\n" for i in range(1, lbuf)]
    else:
        try:  # special handling for class instances
            if not isclass(object) and isclass(type(object)):  # __class__
                file = getfile(module)
                sourcefile = getsourcefile(module)
            else:  # builtins fail with a TypeError
                file = getfile(object)
                sourcefile = getsourcefile(object)
        except (TypeError, AttributeError):  # fail with better error
            file = getfile(object)
            sourcefile = getsourcefile(object)
        if not sourcefile and file[:1] + file[-1:] != "<>":
            raise IOError("source code not available")
        file = sourcefile if sourcefile else file

        module = getmodule(object, file)
        if module:
            lines = linecache.getlines(file, module.__dict__)
        else:
            lines = linecache.getlines(file)

    if not lines:
        raise IOError("could not extract source code")

    # FIXME: all below may fail if exec used (i.e. exec('f = lambda x:x') )
    if ismodule(object):
        return lines, 0

    # NOTE: beneficial if search goes from end to start of buffer history
    name = pat1 = obj = ""
    pat2 = r"^(\s*@)"
    #   pat1b = r'^(\s*%s\W*=)' % name #FIXME: finds 'f = decorate(f)', not exec
    if ismethod(object):
        name = object.__name__
        if name == "<lambda>":
            pat1 = r"(.*(?<!\w)lambda(:|\s))"
        else:
            pat1 = r"^(\s*def\s)"
        object = object.__func__
    if isfunction(object):
        name = object.__name__
        if name == "<lambda>":
            pat1 = r"(.*(?<!\w)lambda(:|\s))"
            obj = object  # XXX: better a copy?
        else:
            pat1 = r"^(\s*def\s)"
        object = object.__code__
    if istraceback(object):
        object = object.tb_frame
    if isframe(object):
        object = object.f_code
    if iscode(object):
        if not hasattr(object, "co_firstlineno"):
            raise IOError("could not find function definition")
        # stdin = object.co_filename == '<stdin>'
        stdin = object.co_filename in ("<console>", "<stdin>")
        # print(f"[dill.source.findsource] object.co_filename: {object.co_filename}, stdin: {stdin}")
        if stdin:
            lnum = len(lines) - 1  # can't get lnum easily, so leverage pat
            if not pat1:
                pat1 = r"^(\s*def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)"
        else:
            lnum = object.co_firstlineno - 1
            pat1 = r"^(\s*def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)"
        pat1 = re.compile(pat1)
        pat2 = re.compile(pat2)
        # XXX: candidate_lnum = [n for n in range(lnum) if pat1.match(lines[n])]
        while lnum > 0:  # XXX: won't find decorators in <stdin> ?
            line = lines[lnum]
            if pat1.match(line):
                if not stdin:
                    break  # co_firstlineno does the job
                if name == "<lambda>":  # hackery needed to confirm a match
                    if _matchlambda(obj, line):
                        break
                else:  # not a lambda, just look for the name
                    if name in line:  # need to check for decorator...
                        hats = 0
                        for _lnum in range(lnum - 1, -1, -1):
                            if pat2.match(lines[_lnum]):
                                hats += 1
                            else:
                                break
                        lnum = lnum - hats
                        break
            lnum = lnum - 1
        return lines, lnum

    try:  # turn instances into classes
        if not isclass(object) and isclass(type(object)):  # __class__
            object = object.__class__  # XXX: sometimes type(class) is better?
            # XXX: we don't find how the instance was built
    except AttributeError:
        pass
    if isclass(object):
        name = object.__name__
        pat = re.compile(r"^(\s*)class\s*" + name + r"\b")
        # make some effort to find the best matching class definition:
        # use the one with the least indentation, which is the one
        # that's most probably not inside a function definition.
        candidates = []
        for i in range(len(lines) - 1, -1, -1):
            match = pat.match(lines[i])
            if match:
                # if it's at toplevel, it's already the best one
                if lines[i][0] == "c":
                    return lines, i
                # else add whitespace to candidate list
                candidates.append((match.group(1), i))
        if candidates:
            # this will sort by whitespace, and by line number,
            # less whitespace first  #XXX: should sort high lnum before low
            candidates.sort()
            return lines, candidates[0][1]
        else:
            raise IOError("could not find class definition")
    raise IOError("could not find code object")


dill.source.findsource = findsource
