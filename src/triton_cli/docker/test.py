class A:
    pass


class B:
    pass


d = {"a": A, "b": B}

c = d.get("c")
if not c:
    print("good")

a = d.get("a")
if not a:
    print("bad")
