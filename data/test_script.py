import os

print(os.listdir())

files = os.listdir()

print(sum([1 if "test_" in f else 0 for f in files]))
