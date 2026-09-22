# Repository Instructions


`scipion-bridge` is an interoperability library for the Scipion framework. It
implements a columnar type system, a declarative streaming framework, and proxies
to manage external files. It integrates with the existing Scipion framework.


The structure of the project:
- `core/struct`: A columnar type system that is internally backed by
numpy/akward array and Apache Arrow
- `core/streaming`: Streaming library built on top of streamz
- `core/typed`: A graph-based type resolution engine to ease the conversion
between different data types. Includes proxy types for passing data to external
programs as files.


The library also includes utilities for wrapping shell commands and automatically
handling reference counting of proxy objects. Different backends are supported
using dependency injection; one of those backends is implemented in `pyworkflow`
for Scipion 3.


## Project Setup
To set up the project install an editable copy of the project in a conda
environment.


To test the library with the `pyworkflow` backend, it needs to be
installed in the Scipion 3 conda environment. This environment is usually called
`scipion3` or similar. Note however that by default Scipion 3 uses Python 3.8
while this library requires at least Python 3.11.


## Verification
Testing is implemented under the directory `test/` and uses pytests for testing.

Verify that the library is correctly typed with pyright.

Use black for formatting.

Use flake8 for static analysis:
```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

## Conventions
`scipion-bridge` is implemented in Python 3.11. When writing code, follow modern
Python patterns.


The code in this library tries to follow modern coding standards like the Google
style guide when possible. The are a few conventions to follow for clean code in
the library.

### Style Notes
#### Format with black, append trailing comma in function calls
Format your code with the Black formatter, and append a leading comma when
calling functions
```
# The lsat comma here after entry.dtype will cause Black to format this
# function call nicely.
buffer = np.zeros((self._capacity, *stacked.shape[1:]), dtype=entry.dtype,)
```


### Avoid overly defensive code
Do not add overly defensive checks in the code. For example avoid:
```
def foo(a):


   bar = getattr(a, "bar", None)
   if bar is not None:
       do_something()
   else:
       some_fallback()
```


Instead, assert the type and access the attribute directly
```
def foo(a):
   assert isinstance(a, MyClassType)
   do_something(a.bar)


```

Code should fail early and consistently.

### Prefer match-case over if-else when possible
Instead of using large if-else blocks, use pattern matching. An example of this
is the __get_item__ method inside sets.


```
def __getitem__( self, key):
   match key:
       case bool():
           ...


       case _ if _is_bool_sequence(key):
           ...


       case _ if _is_int_sequence(key):
           ...
```
This syntax is much more readable than many isinstance(...) calls in nested
conditionals.

### Avoid implicit else branches
If if-else conditional should always be explicit. Avoid creating implicit else
branches by returning from a function that does not run.

```
if is_this_true():
   return compute_result()

return compute_other_result() # This is an implicit else
```

### Avoid deeply nesting conditionals or long functions (Guard Clauses)
Avoid deeply nesting conditionals (never deeper than two levels). Prefer early
returns (guard clauses) to prevent implicit else indentation blocks, keeping the
code flat and readable.

### Never implement special cases in code for unit tests
Unit tests need to test the functionality of the code. Never accomindate unit
tests with special cases. For example, sets and structs are tested using a mock
storage.

**Avoid:**
```
def fetch_backend_data():
    data = ... # get data somehow

    if is_scalar and hasattr(data, item):
        data = data.item()

    # This is extremly bad, because it will create a seperate execution branch
    # for the unit tests specifically, which makes the tests useless! Never do
    # this!
    # 
    # Instead, create a mock type that declares a method item(), and leverage
    # Python's duck typing
```

### Other patterns
- Annotate the code base with types and implement correct overloads
- Maximize comprehensions (list, dict, set) in the code.

### Adhere to the Zen of Python
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!