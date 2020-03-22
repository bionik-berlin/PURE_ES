using DataFrames
using CSV
df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"])
CSV.write("test1.csv", df)
