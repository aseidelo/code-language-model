"""Modelos parametrizados de exercícios, por conceito.

``Function``: a solução é ``body`` e os testes saem de ``reference``, escrita à
parte, para que uma confira a outra. ``$a``, ``$b`` e ``$c`` são os parâmetros
e ``$name`` é o nome da função.
``Module``: a resposta fica em nível de módulo, sem função.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from string import Template


@dataclass
class Function:
    id: str
    concept: str
    names: tuple[str, ...]
    params: tuple[tuple[str, ...], ...]
    en: str
    pt: str
    body: str
    reference: Callable
    inputs: Callable[[Random], tuple]
    edges: tuple[tuple, ...] = ()

    def solution(self, name: str, params: tuple[str, ...]) -> str:
        lines = Template(self.body).substitute(dict(zip("abc", params)), name=name).splitlines()
        return f"def {name}({', '.join(params)}):\n" + "\n".join(f"    {line}" for line in lines)

    def statement(self, nl: str, params: tuple[str, ...]) -> str:
        return Template(getattr(self, nl)).substitute(dict(zip("abc", params)))


@dataclass
class Module:
    id: str
    concept: str
    build: Callable[[Random], tuple[str, str, str, dict]]  # en, pt, solução, valores esperados


WORDS = ("apple", "river", "code", "level", "stone", "music", "noon", "table", "green", "radar", "light", "cloud")
NAMES = ("x", "y", "z", "n", "total", "count", "value", "result", "size", "score", "age", "price")
TEXT_NAMES = ("name", "city", "color", "title", "word", "label")
TWO = (("a", "b"), ("x", "y"), ("first", "second"), ("m", "n"))
ONE = (("n",), ("x",), ("value",), ("number",))
ITEMS = (("items",), ("numbers",), ("values",), ("data",))
TEXT = (("s",), ("text",), ("word",), ("string",))


def num(r: Random) -> int:
    return r.randint(-20, 20)


def pos(r: Random) -> int:
    return r.randint(1, 12)


def nums(r: Random, low: int = 0) -> list[int]:
    return [num(r) for _ in range(r.randint(low, 7))]


def word(r: Random) -> str:
    return r.choice(WORDS)


def phrase(r: Random) -> str:
    return " ".join(r.choices(WORDS, k=r.randint(1, 5)))


def dictionary(r: Random) -> dict[str, int]:
    return {k: r.randint(0, 9) for k in r.sample(WORDS, r.randint(1, 5))}


def spoken(values: list) -> tuple[str, str]:
    if len(values) == 1:
        return repr(values[0]), repr(values[0])
    head = ", ".join(repr(v) for v in values[:-1])
    return f"{head} and {values[-1]!r}", f"{head} e {values[-1]!r}"


def assign_int(r: Random):
    name, value = r.choice(NAMES), r.randint(-50, 100)
    return (f"Define a variable {name} equal to {value}.", f"Defina uma variável {name} igual a {value}.",
            f"{name} = {value}", {name: value})


def assign_str(r: Random):
    name, value = r.choice(TEXT_NAMES), word(r)
    return (f"Set the variable {name} to the string {value!r}.", f"Atribua a string {value!r} à variável {name}.",
            f"{name} = {value!r}", {name: value})


def assign_list(r: Random):
    name, values = r.choice(("items", "numbers", "values", "data")), [r.randint(0, 20) for _ in range(r.randint(2, 5))]
    en, pt = spoken(values)
    return (f"Create a list called {name} containing {en}.", f"Crie uma lista chamada {name} contendo {pt}.",
            f"{name} = {values!r}", {name: values})


def assign_product(r: Random):
    name, a, b = r.choice(NAMES), r.randint(2, 12), r.randint(2, 12)
    return (f"Define a variable {name} with the result of {a} times {b}.",
            f"Defina uma variável {name} com o resultado de {a} vezes {b}.", f"{name} = {a} * {b}", {name: a * b})


def assign_two(r: Random):
    first, second = r.sample(NAMES, 2)
    a, b = num(r), num(r)
    return (f"Define the variable {first} as {a} and the variable {second} as {b}.",
            f"Defina a variável {first} como {a} e a variável {second} como {b}.",
            f"{first} = {a}\n{second} = {b}", {first: a, second: b})


def assign_sum(r: Random):
    first, second, third = r.sample(NAMES, 3)
    a, b = num(r), num(r)
    return (f"Define {first} = {a} and {second} = {b}, then define {third} as their sum.",
            f"Defina {first} = {a} e {second} = {b}, depois defina {third} como a soma dos dois.",
            f"{first} = {a}\n{second} = {b}\n{third} = {first} + {second}", {first: a, second: b, third: a + b})


def assign_dict(r: Random):
    name, key, value = r.choice(("data", "config", "scores")), word(r), r.randint(0, 99)
    return (f"Create a dictionary called {name} that maps the key {key!r} to {value}.",
            f"Crie um dicionário chamado {name} que associa a chave {key!r} a {value}.",
            f"{name} = {{{key!r}: {value}}}", {name: {key: value}})


def is_prime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, n))


def count_chars(s: str) -> dict[str, int]:
    return dict(Counter(s))


def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


TEMPLATES = [
    *[Module(f"atribuicao-{build.__name__}", "atribuicao", build) for build in
      (assign_int, assign_str, assign_list, assign_product, assign_two, assign_sum, assign_dict)],

    Function("add", "expressao", ("add", "add_numbers", "sum_two"), TWO, "returns the sum of $a and $b",
             "retorna a soma de $a e $b", "return $a + $b", lambda a, b: a + b, lambda r: (num(r), num(r))),
    Function("subtract", "expressao", ("subtract", "difference"), TWO, "returns $a minus $b",
             "retorna $a menos $b", "return $a - $b", lambda a, b: a - b, lambda r: (num(r), num(r))),
    Function("multiply", "expressao", ("multiply", "product"), TWO, "returns the product of $a and $b",
             "retorna o produto de $a e $b", "return $a * $b", lambda a, b: a * b, lambda r: (num(r), num(r))),
    Function("double", "expressao", ("double", "twice"), ONE, "returns $a multiplied by 2",
             "retorna $a multiplicado por 2", "return $a * 2", lambda a: a * 2, lambda r: (num(r),)),
    Function("square", "expressao", ("square", "squared"), ONE, "returns the square of $a",
             "retorna o quadrado de $a", "return $a * $a", lambda a: a * a, lambda r: (num(r),)),
    Function("average", "expressao", ("average", "mean_of_two"), TWO, "returns the average of $a and $b",
             "retorna a média de $a e $b", "return ($a + $b) / 2", lambda a, b: (a + b) / 2, lambda r: (num(r), num(r))),
    Function("is_even", "expressao", ("is_even", "even"), ONE, "returns True if $a is even and False otherwise",
             "retorna True se $a for par e False caso contrário", "return $a % 2 == 0", lambda a: a % 2 == 0,
             lambda r: (num(r),), ((2,), (3,))),
    Function("rectangle_area", "expressao", ("rectangle_area", "area"), (("width", "height"), ("w", "h")),
             "returns the area of a rectangle with sides $a and $b", "retorna a área de um retângulo de lados $a e $b",
             "return $a * $b", lambda a, b: a * b, lambda r: (pos(r), pos(r))),
    Function("to_fahrenheit", "expressao", ("to_fahrenheit", "celsius_to_fahrenheit"), (("celsius",), ("c",)),
             "converts $a degrees Celsius to Fahrenheit using the formula c * 9 / 5 + 32",
             "converte $a graus Celsius para Fahrenheit com a fórmula c * 9 / 5 + 32",
             "return $a * 9 / 5 + 32", lambda a: a * 9 / 5 + 32, lambda r: (num(r),)),

    Function("maximum", "condicional", ("maximum", "larger", "max_of_two"), TWO,
             "returns the larger of $a and $b without using max", "retorna o maior entre $a e $b sem usar max",
             "if $a > $b:\n    return $a\nreturn $b", max, lambda r: (num(r), num(r))),
    Function("minimum", "condicional", ("minimum", "smaller", "min_of_two"), TWO,
             "returns the smaller of $a and $b without using min", "retorna o menor entre $a e $b sem usar min",
             "if $a < $b:\n    return $a\nreturn $b", min, lambda r: (num(r), num(r))),
    Function("absolute", "condicional", ("absolute", "abs_value"), ONE,
             "returns the absolute value of $a without using abs", "retorna o valor absoluto de $a sem usar abs",
             "if $a < 0:\n    return -$a\nreturn $a", abs, lambda r: (num(r),), ((-3,), (4,))),
    Function("sign", "condicional", ("sign", "sign_of"), ONE,
             "returns 1 if $a is positive, -1 if it is negative and 0 if it is zero",
             "retorna 1 se $a for positivo, -1 se for negativo e 0 se for zero",
             "if $a > 0:\n    return 1\nif $a < 0:\n    return -1\nreturn 0", lambda a: (a > 0) - (a < 0),
             lambda r: (num(r),), ((0,), (5,), (-2,))),
    Function("is_adult", "condicional", ("is_adult", "can_vote"), (("age",), ("years",)),
             "returns True if $a is at least 18 and False otherwise",
             "retorna True se $a for pelo menos 18 e False caso contrário",
             "if $a >= 18:\n    return True\nreturn False", lambda a: a >= 18, lambda r: (r.randint(1, 60),),
             ((18,), (17,))),
    Function("pass_fail", "condicional", ("pass_fail", "grade"), (("score",), ("points",)),
             "returns 'pass' if $a is at least 60 and 'fail' otherwise",
             "retorna 'pass' se $a for pelo menos 60 e 'fail' caso contrário",
             "if $a >= 60:\n    return 'pass'\nreturn 'fail'", lambda a: "pass" if a >= 60 else "fail",
             lambda r: (r.randint(0, 100),), ((60,), (59,))),
    Function("clamp", "condicional", ("clamp", "limit"), (("x", "low", "high"), ("value", "lower", "upper")),
             "returns $a limited to the range from $b to $c", "retorna $a limitado ao intervalo de $b a $c",
             "if $a < $b:\n    return $b\nif $a > $c:\n    return $c\nreturn $a", lambda a, b, c: min(max(a, b), c),
             lambda r: (num(r), -5, 5), ((-9, -5, 5), (9, -5, 5), (1, -5, 5))),
    Function("fizz", "condicional", ("fizz", "fizz_buzz"), ONE,
             "returns 'FizzBuzz' if $a is divisible by 3 and by 5, 'Fizz' if only by 3, 'Buzz' if only by 5, and $a otherwise",
             "retorna 'FizzBuzz' se $a for divisível por 3 e por 5, 'Fizz' se só por 3, 'Buzz' se só por 5, e $a caso contrário",
             "if $a % 15 == 0:\n    return 'FizzBuzz'\nif $a % 3 == 0:\n    return 'Fizz'\nif $a % 5 == 0:\n    return 'Buzz'\nreturn $a",
             lambda a: "FizzBuzz" if a % 15 == 0 else "Fizz" if a % 3 == 0 else "Buzz" if a % 5 == 0 else a,
             lambda r: (r.randint(1, 40),), ((15,), (9,), (10,), (7,))),

    Function("sum_list", "laco", ("sum_list", "total"), ITEMS, "returns the sum of the numbers in the list $a without using sum",
             "retorna a soma dos números da lista $a sem usar sum",
             "total = 0\nfor item in $a:\n    total += item\nreturn total", sum, lambda r: (nums(r),), (([],),)),
    Function("product_list", "laco", ("product_list", "multiply_all"), ITEMS, "returns the product of the numbers in the list $a",
             "retorna o produto dos números da lista $a",
             "result = 1\nfor item in $a:\n    result *= item\nreturn result",
             lambda a: __import__("math").prod(a), lambda r: ([r.randint(-5, 5) for _ in range(r.randint(1, 5))],),
             (([],), ([2, 3],))),
    Function("count_even", "laco", ("count_even", "even_count"), ITEMS, "returns how many numbers in the list $a are even",
             "retorna quantos números da lista $a são pares",
             "count = 0\nfor item in $a:\n    if item % 2 == 0:\n        count += 1\nreturn count",
             lambda a: sum(x % 2 == 0 for x in a), lambda r: (nums(r),), (([1, 3],),)),
    Function("factorial", "laco", ("factorial", "fact"), ONE, "returns the factorial of $a",
             "retorna o fatorial de $a", "result = 1\nfor i in range(2, $a + 1):\n    result *= i\nreturn result",
             lambda a: __import__("math").factorial(a), lambda r: (r.randint(0, 10),), ((0,), (5,))),
    Function("largest", "laco", ("largest", "find_max"), ITEMS,
             "returns the largest number in the non-empty list $a without using max",
             "retorna o maior número da lista não vazia $a sem usar max",
             "best = $a[0]\nfor item in $a:\n    if item > best:\n        best = item\nreturn best", max,
             lambda r: (nums(r, 1),), (([-5, -2],),)),
    Function("sum_to_n", "laco", ("sum_to_n", "sum_up_to"), ONE, "returns the sum of the integers from 1 to $a",
             "retorna a soma dos inteiros de 1 até $a", "total = 0\nfor i in range(1, $a + 1):\n    total += i\nreturn total",
             lambda a: a * (a + 1) // 2, lambda r: (r.randint(1, 50),)),
    Function("power", "laco", ("power", "raise_to"), (("base", "exponent"), ("b", "e")),
             "returns $a raised to $b using a loop", "retorna $a elevado a $b usando um laço",
             "result = 1\nfor _ in range($b):\n    result *= $a\nreturn result", lambda a, b: a ** b,
             lambda r: (r.randint(-4, 6), r.randint(0, 5)), ((2, 3),)),

    Function("evens", "lista", ("evens", "only_even"), ITEMS, "returns a new list with only the even numbers of $a",
             "retorna uma nova lista só com os números pares de $a", "return [item for item in $a if item % 2 == 0]",
             lambda a: [x for x in a if x % 2 == 0], lambda r: (nums(r),), (([1, 3],),)),
    Function("doubled", "lista", ("doubled", "double_all"), ITEMS, "returns a new list with each number of $a multiplied by 2",
             "retorna uma nova lista com cada número de $a multiplicado por 2", "return [item * 2 for item in $a]",
             lambda a: [x * 2 for x in a], lambda r: (nums(r),), (([],),)),
    Function("positives", "lista", ("positives", "only_positive"), ITEMS, "returns a new list with only the positive numbers of $a",
             "retorna uma nova lista só com os números positivos de $a", "return [item for item in $a if item > 0]",
             lambda a: [x for x in a if x > 0], lambda r: (nums(r),)),
    Function("reverse_list", "lista", ("reverse_list", "reversed_items"), ITEMS, "returns a new list with the elements of $a in reverse order",
             "retorna uma nova lista com os elementos de $a em ordem inversa", "return $a[::-1]",
             lambda a: a[::-1], lambda r: (nums(r),)),
    Function("unique", "lista", ("unique", "remove_duplicates"), ITEMS, "returns a new list with the elements of $a without repeats, keeping their order",
             "retorna uma nova lista com os elementos de $a sem repetições, mantendo a ordem",
             "seen = []\nfor item in $a:\n    if item not in seen:\n        seen.append(item)\nreturn seen",
             lambda a: list(dict.fromkeys(a)), lambda r: ([r.randint(0, 4) for _ in range(r.randint(0, 8))],), (([1, 1, 2],),)),
    Function("first_n", "lista", ("first_n", "take"), (("items", "n"), ("values", "count")),
             "returns the first $b elements of the list $a", "retorna os primeiros $b elementos da lista $a", "return $a[:$b]",
             lambda a, b: a[:b], lambda r: (nums(r), r.randint(0, 4))),
    Function("merge", "lista", ("merge", "concat"), (("first", "second"), ("left", "right")),
             "returns a new list with the elements of $a followed by those of $b",
             "retorna uma nova lista com os elementos de $a seguidos pelos de $b", "return $a + $b",
             lambda a, b: a + b, lambda r: (nums(r), nums(r))),

    Function("reverse_string", "string", ("reverse", "reverse_string"), TEXT, "returns the string $a reversed",
             "retorna a string $a invertida", "return $a[::-1]", lambda a: a[::-1], lambda r: (word(r),), (("",),)),
    Function("count_vowels", "string", ("count_vowels", "vowels"), TEXT, "returns how many vowels the lowercase string $a has",
             "retorna quantas vogais a string minúscula $a tem",
             "count = 0\nfor char in $a:\n    if char in 'aeiou':\n        count += 1\nreturn count",
             lambda a: sum(c in "aeiou" for c in a), lambda r: (word(r),), (("xyz",),)),
    Function("is_palindrome", "string", ("is_palindrome", "palindrome"), TEXT, "returns True if $a reads the same backwards and False otherwise",
             "retorna True se $a for igual lida de trás para frente e False caso contrário", "return $a == $a[::-1]",
             lambda a: a == a[::-1], lambda r: (word(r),), (("aba",), ("ab",))),
    Function("shout", "string", ("shout", "yell"), TEXT, "returns $a in uppercase followed by an exclamation mark",
             "retorna $a em maiúsculas seguido de um ponto de exclamação", "return $a.upper() + '!'",
             lambda a: a.upper() + "!", lambda r: (word(r),)),
    Function("word_count", "string", ("word_count", "count_words"), TEXT, "returns the number of space-separated words in $a",
             "retorna o número de palavras separadas por espaço em $a", "return len($a.split())",
             lambda a: len(a.split()), lambda r: (phrase(r),)),
    Function("initials", "string", ("initials", "acronym"), TEXT, "returns the first letter of each word of $a joined together",
             "retorna a primeira letra de cada palavra de $a, tudo junto", "return ''.join(part[0] for part in $a.split())",
             lambda a: "".join(p[0] for p in a.split()), lambda r: (phrase(r),)),
    Function("count_char", "string", ("count_char", "occurrences"), (("text", "char"), ("s", "c")),
             "returns how many times the character $b appears in $a", "retorna quantas vezes o caractere $b aparece em $a",
             "return $a.count($b)", lambda a, b: a.count(b), lambda r: (phrase(r), r.choice("aeiou"))),
    Function("remove_spaces", "string", ("remove_spaces", "strip_spaces"), TEXT, "returns $a without any spaces",
             "retorna $a sem nenhum espaço", "return $a.replace(' ', '')", lambda a: a.replace(" ", ""), lambda r: (phrase(r),)),

    Function("char_count", "dicionario", ("char_count", "letter_count"), TEXT, "returns a dictionary with how many times each character appears in $a",
             "retorna um dicionário com quantas vezes cada caractere aparece em $a",
             "counts = {}\nfor char in $a:\n    counts[char] = counts.get(char, 0) + 1\nreturn counts",
             count_chars, lambda r: (word(r),), (("",),)),
    Function("invert", "dicionario", ("invert", "swap_keys_values"), (("mapping",), ("data",), ("d",)),
             "returns a new dictionary with the keys and values of $a swapped",
             "retorna um novo dicionário com as chaves e os valores de $a trocados", "return {value: key for key, value in $a.items()}",
             lambda a: {v: k for k, v in a.items()}, lambda r: (dictionary(r),)),
    Function("get_or_default", "dicionario", ("get_or_default", "lookup"), (("mapping", "key"), ("data", "k")),
             "returns the value of $b in the dictionary $a, or 0 if the key does not exist",
             "retorna o valor de $b no dicionário $a, ou 0 se a chave não existir",
             "if $b in $a:\n    return $a[$b]\nreturn 0", lambda a, b: a.get(b, 0), lambda r: (dictionary(r), word(r)),
             (({"a": 5}, "a"), ({"a": 5}, "b"))),
    Function("sum_values", "dicionario", ("sum_values", "total_value"), (("mapping",), ("data",), ("d",)),
             "returns the sum of the values of the dictionary $a", "retorna a soma dos valores do dicionário $a",
             "total = 0\nfor value in $a.values():\n    total += value\nreturn total", lambda a: sum(a.values()),
             lambda r: (dictionary(r),), (({},),)),
    Function("word_lengths", "dicionario", ("word_lengths", "lengths"), (("words",), ("items",)),
             "returns a dictionary that maps each string of the list $a to its length",
             "retorna um dicionário que associa cada string da lista $a ao seu tamanho", "return {item: len(item) for item in $a}",
             lambda a: {w: len(w) for w in a}, lambda r: (r.sample(WORDS, r.randint(0, 4)),)),
    Function("keys_above", "dicionario", ("keys_above", "filter_keys"), (("mapping", "limit"), ("data", "minimum")),
             "returns a list with the keys of $a whose value is greater than $b",
             "retorna uma lista com as chaves de $a cujo valor é maior que $b",
             "return [key for key, value in $a.items() if value > $b]", lambda a, b: [k for k, v in a.items() if v > b],
             lambda r: (dictionary(r), r.randint(0, 8))),

    Function("count_primes", "auxiliar", ("count_primes", "prime_count"), ITEMS,
             "returns how many numbers of $a are prime, using a helper function is_prime",
             "retorna quantos números de $a são primos, usando uma função auxiliar is_prime",
             "def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, n):\n        if n % d == 0:\n            return False\n    return True\n\nreturn sum(1 for item in $a if is_prime(item))",
             lambda a: sum(is_prime(x) for x in a), lambda r: ([r.randint(0, 30) for _ in range(r.randint(0, 7))],), (([4, 6],),)),
    Function("fibonacci", "auxiliar", ("fibonacci", "fib"), ONE, "returns the Fibonacci number at position $a, where position 0 is 0 and position 1 is 1",
             "retorna o número de Fibonacci na posição $a, sendo 0 na posição 0 e 1 na posição 1",
             "if $a < 2:\n    return $a\nreturn $name($a - 1) + $name($a - 2)", fibonacci,
             lambda r: (r.randint(0, 15),), ((0,), (1,), (7,))),
    Function("gcd", "auxiliar", ("gcd",), TWO, "returns the greatest common divisor of the positive integers $a and $b",
             "retorna o máximo divisor comum dos inteiros positivos $a e $b",
             "while $b:\n    $a, $b = $b, $a % $b\nreturn $a", gcd, lambda r: (r.randint(1, 60), r.randint(1, 60)), ((12, 18),)),
    Function("max_of_three", "auxiliar", ("max_of_three", "largest_of_three"), (("a", "b", "c"), ("x", "y", "z")),
             "returns the largest of $a, $b and $c, using a helper function that compares two numbers",
             "retorna o maior entre $a, $b e $c, usando uma função auxiliar que compara dois números",
             "def larger(p, q):\n    if p > q:\n        return p\n    return q\n\nreturn larger(larger($a, $b), $c)", max,
             lambda r: (num(r), num(r), num(r))),
    Function("sum_digits", "auxiliar", ("sum_digits", "digit_sum"), ONE, "returns the sum of the digits of the positive integer $a",
             "retorna a soma dos dígitos do inteiro positivo $a",
             "total = 0\nwhile $a > 0:\n    total += $a % 10\n    $a //= 10\nreturn total",
             lambda a: sum(map(int, str(a))), lambda r: (r.randint(1, 99999),)),
    Function("is_sorted", "auxiliar", ("is_sorted", "sorted_ascending"), ITEMS, "returns True if the list $a is in ascending order and False otherwise",
             "retorna True se a lista $a estiver em ordem crescente e False caso contrário",
             "for i in range(len($a) - 1):\n    if $a[i] > $a[i + 1]:\n        return False\nreturn True",
             lambda a: a == sorted(a), lambda r: (nums(r),), (([1, 2],), ([2, 1],))),
]
