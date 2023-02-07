import pandas

# wget http://www.naturalnumbers.org/P-100000.txt
df = pandas.read_csv("P-100000.txt")
arr = df[" 2"]
lst = list(arr[10_000 < arr][:5_000])
with open("primes.bin", "wb") as f:
    for x in lst:
        f.write(x.to_bytes(8, "little", signed=False))
